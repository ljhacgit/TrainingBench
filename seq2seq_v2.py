"""
Partially use code from,
1. Extensions to TF RNN class by una_dinosaria
2. Julieta Martinez, Javier Romero ( MIT License Copyright (c) 2016 )
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import GRU, Dropout, Dense

---------------------------------------------------------------------------------------------------------------
#  Sequence-to-sequence model for human motion prediction
# -------------------------------------------------------------------------------------------------------------------
class Seq2SeqModel:
    def __init__(self, FLAGES, summaries_dir, dtype=tf.float32):
        self.input_size = FLAGES.input_size
        self.output_size = FLAGES.output_size
        self.datapoint_size = FLAGES.datapoint_size
        self.batch_size = FLAGES.batch_size
        self.unit_size = FLAGES.unit_size
        self.max_gradient_norm = FLAGES.max_gradient_norm
        self.learning_rate = FLAGES.learning_rate
        self.learning_rate_decay_factor = FLAGES.learning_rate_decay_factor
        self.summaries_dir = summaries_dir
        self.residual_connection = FLAGES.residual_connection
        self.sampling_based = FLAGES.sampling_based
        self.attention = FLAGES.attention
        self.reversed_input = FLAGES.reversed_input
        self.dtype = dtype


        # TODO : Summary writers for train and test runs
        # TODO : create seq2seq model


    def encoder(self, enc_input):
        # enc_input : [batch, seq_length, datapoint_size]
        enc_output, enc_state = GRU(self.unit_size, return_sequences=True, return_state=True)(enc_input)
        return enc_output, enc_state

    def decoder(self, enc_state, dec_input):
        dec_output = GRU(self.unit_size)(dec_input, initial_state=enc_state)
        dec_dense = Dense(self.datapoint_size)(dec_output)
        return dec_dense

    # TODO : Create Attention
    def attention(self):
        pass

    # TODO : Create Residual Connection

    # TODO : Create Samlping Based Loss

    # TODO : Create reversed input

    def step(self, enc_input, dec_input, target, is_test):
        if is_test:

            return step_loss, step_output, loss_summary
        else:

            return step_loss
