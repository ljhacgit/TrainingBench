"""
Partially use code from,
Julieta Martinez, Javier Romero ( MIT License Copyright (c) 2016 )
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

import data_utils_v2
import forward_kinematics

# ---------------------------------------------------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------------------------------------------------
# seq2seq model parameter
seq2seq_datapoint_size = 99  # Data Per Frame of data
seq2seq_train_datapoint_size = 54  # Data Per Frame of data
seq2seq_unit_size = 1024  # Size of each rnn model unit.  
seq2seq_input_size = 26  # Number of frames to feed into the encoder. 25 fps  
seq2seq_output_size = 26  # Number of frames that the decoder has to predict. 25fps  

# seq2seq model hyper-parameter : architecture
seq2seq_residual_connection = False # Add a residual connection that effectively models velocities
seq2seq_sampling_based = False  # Use Sampling Based loss : CANNOT USE WITH ATTENTION   
seq2seq_attention = True  # Use Attention Mechanism  
seq2seq_reversed_input = False  # Use reversed input data  

# seq2seq model hyper-parameter : values
seq2seq_learning_rate = .005  # Learning rate.  
seq2seq_learning_rate_decay_factor = 0.95 # Learning rate is multiplied by this much. 1 means no decay.  
seq2seq_learning_rate_decay_step = 2500  # Every this many steps, do decay.  
seq2seq_max_gradient_norm = 5  # Clip gradients to this norm.  

# seq2seq basic train parameter
train_batch_size = 16  # Batch size to use during training.  
train_iterations = 100  # Iterations to train for.  
train_test_every = 50  # How often to compute error on the test set.  
train_save_every = 50  # How often to compute error on the test set.  
train_checkpoint_load = 0  # Weather to load checkpoint or not.  
train_sample = False  # Set to True for sampling.  

# additional
use_cpu = False  # Whether to use the CPU  

# h36m data parameter
action = "walking"  # The action to train on. all means all the actions all_periodic means walking, eating and smoking  
data_dir = os.path.normpath("./h3.6m/dataset")  # Data directory  
train_dir = os.path.normpath("./experiments")  # Training directory.  

# train, summaries data path
train_dir = os.path.normpath(os.path.join(train_dir, 'action;{0}_'.format(  action) + 'out;{0}_'.format(
      seq2seq_output_size) + 'iterations;{0}_'.format(train_iterations) + 'size;{0}_'.format(
      seq2seq_unit_size) + 'lr;{0}_'.format(  seq2seq_learning_rate) + 'samplingBased;{0}_'.format(
      seq2seq_sampling_based) + 'attention;{0}_'.format(seq2seq_attention) + 'reversedInput;{0}_'.format(
      seq2seq_reversed_input) + 'residual_connection;{0}'.format(seq2seq_residual_connection)))
summaries_dir = os.path.normpath(os.path.join(train_dir, "log"))
visualize_dir = os.path.join(train_dir, "visual_data.h5")


# ---------------------------------------------------------------------------------------------------------------------
# Utils, data related utils are in data_util.py
# ---------------------------------------------------------------------------------------------------------------------
def define_actions(action):
    actions = ["walking", "eating", "smoking", "discussion", "directions", "greeting", "phoning", "posing", "purchases",
               "sitting", "sittingdown", "takingphoto", "waiting", "walkingdog", "walkingtogether"]

    if action in actions:
        return [action]
    if action == "all":
        return actions
    raise (ValueError, "Unrecognized action: %d" % action)


# -------------------------------------------------------------------------------------------------------------------
#  Sequence-to-sequence model for human motion prediction
# -------------------------------------------------------------------------------------------------------------------
# decoder layer
class LinearSpaceDecoder(keras.layers.Layer):
    def __init__(self, units):
        super(LinearSpaceDecoder, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


# print training info
print()
print("Start train with parameter as,")
print(" train_batch_size         : %d" % train_batch_size)
print(" train_iterations         : %d" % train_iterations)
print()

# save training info
with open(train_dir + "/train_log.txt", "a") as log_file:
    log_file.write("\n")
    log_file.write("Start train with parameter as,\n")
    log_file.write(" train_batch_size         : %d\n" % train_batch_size)
    log_file.write(" train_iterations         : %d\n" % train_iterations)
    log_file.write("\n")

# get data
actions = define_actions(action)
# train_data, test_data : {(subject, action, subaction):[n, d]}, normalize_parameter : {parameter_name : data}
train_data, normalize_parameter \
    = data_utils_v2.get_preprocessed_data(actions, seq2seq_input_size, seq2seq_output_size, data_dir)

enc_input_data, dec_input_data, target_data = data_utils_v2.get_seq2seq_batch(train_data, train_batch_size,
                                                                              seq2seq_input_size,
                                                                              seq2seq_output_size,
                                                                              seq2seq_train_datapoint_size)
# print(enc_input_data.shape)

# init data
enc_input = keras.Input(shape=(seq2seq_input_size, seq2seq_train_datapoint_size))
dec_input = keras.Input(shape=(seq2seq_output_size, seq2seq_train_datapoint_size))
target = keras.Input(shape=(seq2seq_output_size, seq2seq_train_datapoint_size))

# encoder, decoder init, calculate data
# enc
enc_output, enc_state = keras.layers.GRU(seq2seq_unit_size, return_state=True, return_sequences=True)(enc_input)
# dec
dec_output = keras.layers.GRU(seq2seq_unit_size, return_sequences=True)(dec_input, initial_state=enc_state)
# reshape
dec_output = LinearSpaceDecoder(seq2seq_train_datapoint_size)(dec_output)

# create model
model = keras.Model([enc_input, dec_input], dec_output)
print(model.summary())

# compile model
model.compile(
    optimizer="sgd",
    loss="mse",
    metrics=[tf.keras.metrics.MeanSquaredError()]
)

# train model
model.fit(
    [enc_input_data, dec_input_data],
    target_data,
    batch_size=train_batch_size,
    epochs=train_iterations,
    validation_split=0.2
)
# save model
model.save("s2s")
