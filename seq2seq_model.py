"""
Partially use code from,
1. Extensions to TF RNN class by una_dinosaria
2. Julieta Martinez, Javier Romero ( MIT License Copyright (c) 2016 )
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import RNNCell
import numpy as np

import os

# The import for LSTMStateTuple changes in TF >= 1.2.0
from pkg_resources import parse_version as pv


# -------------------------------------------------------------------------------------------------------------------
#  Residual Connection For RNN
# -------------------------------------------------------------------------------------------------------------------
class ResidualWrapper(RNNCell):
    def __init__(self, cell):
        """
        Create a cell with added residual connection.
        Args:
          cell: an RNNCell. The input is added to the output.
        Raises:
          TypeError: if cell is not an RNNCell.
        """

        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")

        self._cell = cell

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        """Run the cell and add a residual connection."""

        # Run the rnn as usual
        output, new_state = self._cell(inputs, state, scope)

        # Add the residual connection
        output = tf.add(output, inputs)

        return output, new_state


# -------------------------------------------------------------------------------------------------------------------
#  Linear Space Decoder For RNN
# -------------------------------------------------------------------------------------------------------------------
class LinearSpaceDecoderWrapper(RNNCell):
    def __init__(self, cell, output_size):
        """
        Create a cell with with a linear encoder in space.

        Args:
          cell: an RNNCell. The input is passed through a linear layer.
        Raises:
          TypeError: if cell is not an RNNCell.
        """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")

        # this need super class __init__, see more details in fallow link
        # https://www.tensorflow.org/tutorials/customization/custom_layers
        super(LinearSpaceDecoderWrapper, self).__init__()

        self._cell = cell

        in_size = self._cell.state_size
        self.w_out = tf.get_variable("w_out",
                                     [in_size, output_size],
                                     dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))
        self.b_out = tf.get_variable("b_out", [output_size],
                                     dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

        self.linear_output_size = output_size

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self.linear_output_size

    def __call__(self, inputs, state, scope=None):
        """Use a linear layer and pass the output to the cell."""

        # Run the rnn as usual
        output, new_state = self._cell(inputs, state, scope)

        # Apply the multiplication to everything
        output = tf.matmul(output, self.w_out) + self.b_out

        return output, new_state


# -------------------------------------------------------------------------------------------------------------------
#  Sequence-to-sequence model for human motion prediction
# -------------------------------------------------------------------------------------------------------------------
class Seq2SeqModel:
    def __init__(self, input_size, output_size, datapoint_size, batch_size, unit_size,
                 max_gradient_norm, learning_rate, learning_rate_decay_factor, summaries_dir,
                 residual_connection, sampling_based, attention, reversed_input, dtype=tf.float32):

        # TODO : Maybe use tf.FLAGS to sand parameter?
        self.input_size = input_size
        self.output_size = output_size
        self.datapoint_size = datapoint_size
        self.batch_size = batch_size
        self.unit_size = unit_size
        self.max_gradient_norm = max_gradient_norm
        self.learning_rate = learning_rate
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.summaries_dir = summaries_dir
        self.residual_connection = residual_connection
        self.sampling_based = sampling_based
        self.attention = attention
        self.reversed_input = reversed_input
        self.dtype = dtype

        # Summary writers for train and test runs
        self.train_writer = tf.summary.FileWriter(os.path.normpath(os.path.join(summaries_dir, 'train')))
        self.test_writer = tf.summary.FileWriter(os.path.normpath(os.path.join(summaries_dir, 'test')))
        # Init data
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        # create seq2seq model
        self.enc_in, self.dec_in, self.dec_out = self.init_input_placeholder()
        enc_in_transformed, dec_in_transformed, dec_out_transformed \
            = self.transform_input(self.enc_in, self.dec_in, self.dec_out)

        with vs.variable_scope("seq2seq", reuse=tf.AUTO_REUSE):
            # seq2seq encoder
            self.enc_output, enc_states = self.seq2seq_encoder(enc_in_transformed)
            # seq2seq decoder
            self.outputs = self.seq2seq_decoder(dec_in_transformed, self.enc_output, enc_states)

        # loss and other operations
        self.loss, self.loss_summary, self.gradient_norms, self.updates, self.learning_rate_summary, self.saver \
            = self.init_operations(dec_out_transformed, self.outputs)

    def init_input_placeholder(self):
        # Transform the inputs
        with tf.name_scope("inputs"):
            enc_in = tf.placeholder(self.dtype, shape=[self.batch_size, self.input_size - 1, self.datapoint_size],
                                    name="enc_in")
            dec_in = tf.placeholder(self.dtype, shape=[self.batch_size, self.output_size, self.datapoint_size],
                                    name="dec_in")
            dec_out = tf.placeholder(self.dtype, shape=[self.batch_size, self.output_size, self.datapoint_size],
                                     name="dec_out")

        # data shaped as [batch, frame, datapoint]
        return enc_in, dec_in, dec_out

    def transform_input(self, enc_in, dec_in, dec_out):
        with tf.name_scope("inputs"):
            if self.reversed_input:
                enc_in = tf.reverse(enc_in, [1])

            # if data needs additional processing you can use this part to add what you need
            if self.sampling_based:
                # [batch, frame, datapoint] -> [frame, batch, datapoint]
                enc_in = tf.transpose(enc_in, [1, 0, 2])
                dec_in = tf.transpose(dec_in, [1, 0, 2])
                dec_out = tf.transpose(dec_out, [1, 0, 2])

                # [frame, batch, datapoint] -> [-1, datapoint]
                enc_in = tf.reshape(enc_in, [-1, self.datapoint_size])
                dec_in = tf.reshape(dec_in, [-1, self.datapoint_size])
                dec_out = tf.reshape(dec_out, [-1, self.datapoint_size])

                # [-1, datapoint] -> [[batch, datapoint], [batch, datapoint], ...] (len = frame)
                enc_in = tf.split(enc_in, self.input_size - 1, axis=0)
                dec_in = tf.split(dec_in, self.output_size, axis=0)
                dec_out = tf.split(dec_out, self.output_size, axis=0)

        return enc_in, dec_in, dec_out

    def init_operations(self, dec_out, outputs):
        # loss 계산
        with tf.name_scope("loss_angles"):
            loss_angles = tf.reduce_mean(tf.square(tf.subtract(dec_out, outputs)))
        loss = loss_angles
        loss_summary = tf.summary.scalar('loss/loss', loss)

        # 경사하강법 관련, 파라미터 설정, 업데이터 오퍼레이션 설정
        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)

        # 기울기 클립핑 관련
        # tf.gradients : 모든 학습 가능한 값에 대해서 d(loss)/d(val)을 계산해 tensor 를 반환
        gradients = tf.gradients(loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        gradient_norms = norm
        updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        # 텐서보드 관련
        # Keep track of the learning rate
        learning_rate_summary = tf.summary.scalar('learning_rate/learning_rate', self.learning_rate)

        # 세이버 설정
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        return loss, loss_summary, gradient_norms, updates, learning_rate_summary, saver

    def seq2seq_encoder(self, enc_in):
        # create GRU cell
        enc_cell = tf.contrib.rnn.GRUCell(self.unit_size, name="enc_cell")

        if self.attention:  # attention
            if self.sampling_based:  # attention, sampling_based
                raise Exception("Cannot use sampling_based with attention")
            else:  # attention
                return tf.nn.dynamic_rnn(enc_cell, enc_in, dtype=self.dtype)
        else:
            if self.sampling_based:  # sampling_based
                enc_cell = LinearSpaceDecoderWrapper(enc_cell, self.datapoint_size)
                return tf.contrib.rnn.static_rnn(enc_cell, enc_in, dtype=self.dtype)
            else:  # nothing and ...
                return tf.nn.dynamic_rnn(enc_cell, enc_in, dtype=self.dtype)

    def seq2seq_decoder(self, dec_in, enc_output, enc_states):
        # crate GRU cell
        dec_cell = tf.contrib.rnn.GRUCell(self.unit_size, name="dec_cell")

        # create initial_state
        initial_state = enc_states
        if self.attention:  # attention
            if self.sampling_based:  # attention, sampling_based
                raise Exception("Cannot use sampling_based with attention")
            else:  # attention
                attn_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.unit_size, memory=enc_output)
                initial_state = dec_cell.zero_state(dtype=self.dtype, batch_size=self.batch_size)
                dec_cell = tf.contrib.seq2seq.AttentionWrapper(cell=dec_cell, attention_mechanism=attn_mechanism,
                                                               output_attention=False)
                dec_cell = LinearSpaceDecoderWrapper(dec_cell, self.datapoint_size)
                if self.residual_connection:  # attention, residual_connection
                    dec_cell = ResidualWrapper(dec_cell)
                outputs, _ = tf.nn.dynamic_rnn(dec_cell, dec_in, initial_state=initial_state, dtype=self.dtype)
                return outputs

        else:  # nothing
            dec_cell = LinearSpaceDecoderWrapper(dec_cell, self.datapoint_size)
            if self.residual_connection:  # residual_connection
                dec_cell = ResidualWrapper(dec_cell)
            if self.sampling_based:  # sampling_based
                def lf(prev, i):
                    return prev

                outputs, _ = tf.contrib.legacy_seq2seq.rnn_decoder(dec_in, initial_state, dec_cell, loop_function=lf)
                return outputs
            else:  # nothing
                outputs, _ = tf.nn.dynamic_rnn(dec_cell, dec_in, initial_state=initial_state, dtype=self.dtype)
                return outputs

    def step(self, session, encoder_inputs, decoder_inputs, target, forward_only):
        # 각각 입력 데이터 전처리 및 placeholder
        input_feed = {self.enc_in: encoder_inputs,
                      self.dec_in: decoder_inputs,
                      self.dec_out: target}

        # 순전파, 역전파 계산 - 학습용
        if not forward_only:
            # Training step
            output_feed = [self.updates,  # Update Op that does SGD.
                           self.gradient_norms,  # Gradient norm.
                           self.loss,
                           self.loss_summary,
                           self.learning_rate_summary]

            # 입력되는 placeholder에 대해 출력값이 나타나도록 되어 있는 형태
            outputs = session.run(output_feed, input_feed)
            return outputs[1], outputs[2], outputs[3], outputs[4]  # Gradient norm, loss, summaries

        # 순전파 계산 - 테스트용
        else:
            # Validation step
            output_feed = [self.loss,  # Loss for this batch.
                           self.outputs,
                           self.loss_summary]

            outputs = session.run(output_feed, input_feed)

            step_output = outputs[1]
            if self.sampling_based:
                seq_length = len(step_output)
                batch_size, datapoint_size = step_output[0].shape
                # [seq_length, batch_size, datapoint]
                step_output = np.concatenate(step_output)
                step_output = np.reshape(step_output, (seq_length, batch_size, datapoint_size))
                step_output = np.transpose(step_output, [1, 0, 2])
                # [batch_size, seq_length, datapoint]

            return outputs[0], step_output, outputs[2]  # No gradient norm
