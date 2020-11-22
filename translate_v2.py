"""
Partially use code from,
Julieta Martinez, Javier Romero ( MIT License Copyright (c) 2016 )
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import matplotlib.pyplot as plt

import data_utils
import seq2seq_v2
import forward_kinematics

# ---------------------------------------------------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------------------------------------------------
# seq2seq model parameter
tf.app.flags.DEFINE_integer("seq2seq_datapoint_size", 99, "Data Per Frame of data")
tf.app.flags.DEFINE_integer("seq2seq_train_datapoint_size", 54, "Data Per Frame of data")
tf.app.flags.DEFINE_integer("seq2seq_unit_size", 1024, "Size of each rnn model unit.")
tf.app.flags.DEFINE_integer("seq2seq_input_size", 26, "Number of frames to feed into the encoder. 25 fps")
tf.app.flags.DEFINE_integer("seq2seq_output_size", 26, "Number of frames that the decoder has to predict. 25fps")

# seq2seq model hyper-parameter : architecture
tf.app.flags.DEFINE_boolean("seq2seq_residual_connection", False,
                            "Add a residual connection that effectively models velocities")
tf.app.flags.DEFINE_boolean("seq2seq_sampling_based", False, "Use Sampling Based loss : CANNOT USE WITH ATTENTION ")
tf.app.flags.DEFINE_boolean("seq2seq_attention", True, "Use Attention Mechanism")
tf.app.flags.DEFINE_boolean("seq2seq_reversed_input", False, "Use reversed input data")

# seq2seq model hyper-parameter : values
tf.app.flags.DEFINE_float("seq2seq_learning_rate", .005, "Learning rate.")
tf.app.flags.DEFINE_float("seq2seq_learning_rate_decay_factor", 0.95,
                          "Learning rate is multiplied by this much. 1 means no decay.")
tf.app.flags.DEFINE_integer("seq2seq_learning_rate_decay_step", 2500, "Every this many steps, do decay.")
tf.app.flags.DEFINE_float("seq2seq_max_gradient_norm", 5, "Clip gradients to this norm.")

# seq2seq basic train parameter
tf.app.flags.DEFINE_integer("train_batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("train_iterations", 100, "Iterations to train for.")
tf.app.flags.DEFINE_integer("train_test_every", 50, "How often to compute error on the test set.")
tf.app.flags.DEFINE_integer("train_save_every", 50, "How often to compute error on the test set.")
tf.app.flags.DEFINE_boolean("train_checkpoint_load", 0, "Weather to load checkpoint or not.")
tf.app.flags.DEFINE_boolean("train_sample", False, "Set to True for sampling.")

# additional
tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")

# h36m data parameter
tf.app.flags.DEFINE_string("action", "walking", "The action to train on. all means all the actions, \
                                                                      all_periodic means walking, eating and smoking")
tf.app.flags.DEFINE_string("data_dir", os.path.normpath("./h3.6m/dataset"), "Data directory")
tf.app.flags.DEFINE_string("train_dir", os.path.normpath("./experiments"), "Training directory.")

# def FLAGS
FLAGS = tf.app.flags.FLAGS

# train, summaries data path
train_dir = os.path.normpath(os.path.join(FLAGS.train_dir, 'action;{0}_'.format(FLAGS.action) +
    'out;{0}_'.format(FLAGS.seq2seq_output_size) + 'iterations;{0}_'.format(FLAGS.train_iterations) + \
    'size;{0}_'.format(FLAGS.seq2seq_unit_size) + 'lr;{0}_'.format(FLAGS.seq2seq_learning_rate) + \
    'samplingBased;{0}_'.format(FLAGS.seq2seq_sampling_based) + 'attention;{0}_'.format(FLAGS.seq2seq_attention) + \
    'reversedInput;{0}_'.format(FLAGS.seq2seq_reversed_input) + \
    'residual_connection;{0}'.format(FLAGS.seq2seq_residual_connection)))
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


# ---------------------------------------------------------------------------------------------------------------------
# Train and evaluate
# ---------------------------------------------------------------------------------------------------------------------
def init_seq2seq_model():
    model = seq2seq_v2.Seq2SeqModel(FLAGS, summaries_dir)

    # TODO : Build Restore Function


    # print model information
    print()
    print("Create Model with Parameter as,")
    print(" input_size          : %d" % FLAGS.seq2seq_input_size)
    print(" output_size         : %d" % FLAGS.seq2seq_output_size)
    print(" learning_rate       : %d" % FLAGS.seq2seq_learning_rate)
    print(" residual_connection : %d" % FLAGS.seq2seq_residual_connection)
    print(" sampling_based      : %d" % FLAGS.seq2seq_sampling_based)
    print(" attention           : %d" % FLAGS.seq2seq_attention)
    print(" reversed_input      : %d" % FLAGS.seq2seq_reversed_input)
    print()

    print("Model created")
    print()

    # save model information
    with open(train_dir + "/train_log.txt", "a") as log_file:
        log_file.write("\n")
        log_file.write("Create Model with Parameter as,\n")
        log_file.write(" input_size          : %d\n" % FLAGS.seq2seq_input_size)
        log_file.write(" output_size         : %d\n" % FLAGS.seq2seq_output_size)
        log_file.write(" learning_rate       : %d\n" % FLAGS.seq2seq_learning_rate)
        log_file.write(" residual_connection : %d\n" % FLAGS.seq2seq_residual_connection)
        log_file.write(" sampling_based      : %d\n" % FLAGS.seq2seq_sampling_based)
        log_file.write(" attention           : %d\n" % FLAGS.seq2seq_attention)
        log_file.write(" reversed_input      : %d\n\n" % FLAGS.seq2seq_reversed_input)

        log_file.write("Model created\n")
        log_file.write("\n")

    return model


def train():
    # get data
    actions = define_actions(FLAGS.action)

    # train_data, test_data : {(subject, action, subaction):[n, d]}, normalize_parameter : {parameter_name : data}
    train_data, test_data, normalize_parameter = data_utils.get_preprocessed_data(actions, FLAGS.seq2seq_input_size,
                                                                                  FLAGS.seq2seq_output_size,
                                                                                  FLAGS.data_dir)
    # init seq2se2 model
    model = init_seq2seq_model()

    # print training info
    print()
    print("Start train with parameter as,")
    print(" train_batch_size         : %d" % FLAGS.train_batch_size)
    print(" train_iterations         : %d" % FLAGS.train_iterations)
    print()

    # save training info
    with open(train_dir + "/train_log.txt", "a") as log_file:
        log_file.write("\n")
        log_file.write("Start train with parameter as,\n")
        log_file.write(" train_batch_size         : %d\n" % FLAGS.train_batch_size)
        log_file.write(" train_iterations         : %d\n" % FLAGS.train_iterations)
        log_file.write("\n")

    # TODO : Do Training
    # TODO : Visualize Data
    # TODO : Save Model

