"""
Partially use code from,
Julieta Martinez, Javier Romero ( MIT License Copyright (c) 2016 )
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import matplotlib.pyplot as plt

import data_utils
import seq2seq_model
import forward_kinematics

# ---------------------------------------------------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------------------------------------------------
# for campatibility
tf.reset_default_graph()
for keys in [keys for keys in tf.app.flags.FLAGS._flags()]:
  tf.app.flags.FLAGS.__delattr__(keys)

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
tf.app.flags.DEFINE_boolean("seq2seq_attention", False, "Use Attention Mechanism")
tf.app.flags.DEFINE_boolean("seq2seq_reversed_input", False, "Use reversed input data")

# seq2seq model hyper-parameter : values
tf.app.flags.DEFINE_float("seq2seq_learning_rate", .005, "Learning rate.")
tf.app.flags.DEFINE_float("seq2seq_learning_rate_decay_factor", 0.95,
                          "Learning rate is multiplied by this much. 1 means no decay.")
tf.app.flags.DEFINE_integer("seq2seq_learning_rate_decay_step", 2500, "Every this many steps, do decay.")
tf.app.flags.DEFINE_float("seq2seq_max_gradient_norm", 5, "Clip gradients to this norm.")

# seq2seq basic train parameter
tf.app.flags.DEFINE_integer("train_batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("train_iterations", 50000, "Iterations to train for.")
tf.app.flags.DEFINE_integer("train_test_every", 1000, "How often to compute error on the test set.")
tf.app.flags.DEFINE_integer("train_save_every", 1000, "How often to compute error on the test set.")
tf.app.flags.DEFINE_boolean("train_checkpoint_load", 0, "Weather to load checkpoint or not.")
tf.app.flags.DEFINE_boolean("train_sample", False, "Set to True for sampling.")

# additional
tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")

# h36m data parameter
tf.app.flags.DEFINE_string("action", "walking", "The action to train on. all means all the actions, \
                                                                      all_periodic means walking, eating and smoking")
tf.app.flags.DEFINE_string("data_dir", os.path.normpath("/data/h3.6m/dataset"), "Data directory")
tf.app.flags.DEFINE_string("train_dir", os.path.normpath("/experiments"), "Training directory.")

# def FLAGS
FLAGS = tf.app.flags.FLAGS

# for campatibility
remaining_args = FLAGS([sys.argv[0]] + [flag for flag in sys.argv if flag.startswith("--")])
assert(remaining_args == [sys.argv[0]])

# train, summaries data path
train_dir = os.path.normpath(os.path.join(FLAGS.train_dir, 'action;{0}_'.format(FLAGS.action)+
            'out;{0}_'.format(FLAGS.seq2seq_output_size) + 'iterations;{0}_'.format(FLAGS.train_iterations) + \
            'size;{0}_'.format(FLAGS.seq2seq_unit_size) + 'lr;{0}_'.format(FLAGS.seq2seq_learning_rate) + \
            'samplingBased;{0}_'.format(FLAGS.seq2seq_sampling_based) + \
            'attention;{0}_'.format(FLAGS.seq2seq_attention) + \
            'reversedInput;{0}_'.format(FLAGS.seq2seq_reversed_input) +\
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

  if action == "all_srnn":
    return ["walking", "eating", "smoking", "discussion"]

  raise (ValueError, "Unrecognized action: %d" % action)


# ---------------------------------------------------------------------------------------------------------------------
# Train and evaluate
# ---------------------------------------------------------------------------------------------------------------------
def init_seq2seq_model():
  # Limit TF to take a fraction of the GPU memory
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}

  # create tensorflow session
  sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count=device_count))

  model = seq2seq_model.Seq2SeqModel(
    FLAGS.seq2seq_input_size,
    FLAGS.seq2seq_output_size,
    FLAGS.seq2seq_train_datapoint_size,
    FLAGS.train_batch_size,
    FLAGS.seq2seq_unit_size,
    FLAGS.seq2seq_max_gradient_norm,
    FLAGS.seq2seq_learning_rate,
    FLAGS.seq2seq_learning_rate_decay_factor,
    summaries_dir,
    FLAGS.seq2seq_residual_connection,
    FLAGS.seq2seq_sampling_based,
    FLAGS.seq2seq_attention,
    FLAGS.seq2seq_reversed_input)

  # 전체 파라미터 초기화
  if not FLAGS.train_checkpoint_load:
    sess.run(tf.global_variables_initializer())
  else:
    print("Checkpoint Loaded")
    model.saver.restore(sess, data_utils.get_checkpoint(train_dir))

  # 계산 그래프를 출력을 위해 summary 에 저장
  # ref) self.train_writer = tf.summary.FileWriter(os.path.normpath(os.path.join(summaries_dir, 'train')))
  model.train_writer.add_graph(sess.graph)

  # print model infomation
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

  return sess, model


def train():
  # get data
  actions = define_actions(FLAGS.action)
  # train_data, test_data : {(subject, action, subaction):[n, d]}, normalize_parameter : {parameter_name : data}
  train_data, test_data, normalize_parameter = data_utils.get_preprocessed_data(actions, FLAGS.seq2seq_input_size,
                                                           FLAGS.seq2seq_output_size, FLAGS.data_dir)

  # init seq2se2 model
  sess, model = init_seq2seq_model()

  print()
  print("Start train with parameter as,")
  print(" train_batch_size         : %d" % FLAGS.train_batch_size)
  print(" train_iterations         : %d" % FLAGS.train_iterations)
  print()

  with open(train_dir + "/train_log.txt", "a") as log_file:
    log_file.write("\n")
    log_file.write("Start train with parameter as,\n")
    log_file.write(" train_batch_size         : %d\n" % FLAGS.train_batch_size)
    log_file.write(" train_iterations         : %d\n" % FLAGS.train_iterations)
    log_file.write("\n")

  # init sess
  with sess:
    # This is the training loop
    current_step = 0
    previous_losses = []
    step_time, step_avg_loss = 0, 0

    # 반복 횟수에 대해
    for _ in xrange(FLAGS.train_iterations):
      # 시간 측정
      start_time = time.time()

      # 배치를 가져와서
      # encoder_inputs, decoder_inputs, decoder_outputs : [batch, seq_length, datapoint_size]
      encoder_inputs, decoder_inputs, decoder_outputs = data_utils.get_seq2seq_batch(train_data,
          FLAGS.train_batch_size, FLAGS.seq2seq_input_size, FLAGS.seq2seq_output_size, FLAGS.seq2seq_train_datapoint_size)
      # step 진행, 각 스텝에서 배치 데이터에 대해 신경망을 돌리고, 최적화를 진행
      _, step_loss, loss_summary, lr_summary = model.step(sess, encoder_inputs, decoder_inputs, decoder_outputs, False)

      # summary 에 loss, learning_rate 저장
      model.train_writer.add_summary(loss_summary, current_step)
      model.train_writer.add_summary(lr_summary, current_step)

      # 매 10 step 마다 loss 출력
      if current_step % 10 == 0:
        print("step {0:04d}; step_loss: {1:.4f}".format(current_step, step_loss))

      # 시간 측정
      step_time += (time.time() - start_time) / FLAGS.train_test_every

      # step_avg_loss 의 현 배치의 평균을 업데이트
      step_avg_loss += step_loss / FLAGS.train_test_every

      # 다음 스텝으로
      current_step += 1

      # learning_rate decay 값 적용
      if current_step % FLAGS.seq2seq_learning_rate_decay_step == 0:
        sess.run(model.learning_rate_decay_op)

      # 체크포인크 저장, 각종 값 출력
      # Once in a while, we save checkpoint, print statistics, and run evaluates.
      if current_step % FLAGS.train_test_every == 0:

        # 이 부분에서 현재 상태에 대한 test step 진행, step 에서 순전파에 대한 데이터만을 받아온다, 구체적 내용은 step 주석 참고
        # encoder_inputs, decoder_inputs, decoder_outputs : [batch, seq_length, datapoint_size]
        encoder_inputs, decoder_inputs, decoder_outputs = data_utils.get_seq2seq_batch(test_data,
        FLAGS.train_batch_size, FLAGS.seq2seq_input_size, FLAGS.seq2seq_output_size, FLAGS.seq2seq_train_datapoint_size)
        
        step_loss, step_output, loss_summary = model.step(sess, encoder_inputs, decoder_inputs, decoder_outputs, True)

        # Loss book-keeping
        test_step_loss = step_loss
        model.test_writer.add_summary(loss_summary, current_step)

        # get loss as euler angle
        test_step_losses, test_step_euler_losses = data_utils.get_loss(step_output, decoder_outputs, normalize_parameter)
        test_step_euler_loss = float(np.mean(test_step_euler_losses))

        # print data
        print("\nData Logger")
        print("RAW DATA")
        plt.plot(test_step_euler_losses, label="euler")
        plt.legend()
        plt.show()
        plt.plot(test_step_losses, label="expmap")
        plt.legend()
        plt.show()
        print("--------------------------")
        print("{0: <16} |".format("data_millisec"), end="")
        for ms in [80, 160, 320, 400, 560, 1000]:
          print(" {0:5d} |".format(ms), end="")
        print()
        print("{0: <16} |".format("error_expmap "), end="")
        for ms in [1, 3, 7, 9, 13, 24]:
          if FLAGS.seq2seq_output_size >= ms + 1:
            print(" {0:.3f} |".format(test_step_losses[ms]), end="")
          else:
            print("   n/a |", end="")
        print()
        print("{0: <16} |".format("error_euler "), end="")
        for ms in [1, 3, 7, 9, 13, 24]:
          if FLAGS.seq2seq_output_size >= ms + 1:
            print(" {0:.3f} |".format(test_step_euler_losses[ms]), end="")
          else:
            print("   n/a |", end="")
        print("\n--------------------------")
        print("Global step:         %d\n"
              "Learning rate:       %.4f\n"
              "Step-time (ms):     %.4f\n"
              "Train step loss avg (expmap):      %.4f\n"
              "--------------------------\n"
              "Test step loss (expmap):            %.4f\n"
              "Test step loss (euler angle):       %.4f"
        % (model.global_step.eval(),model.learning_rate.eval(), step_time * 1000, step_avg_loss,
                                                                        test_step_loss, test_step_euler_loss))
        print("--------------------------\n")

        previous_losses.append(step_avg_loss)

        # data save
        if current_step % FLAGS.train_save_every == 0:
          print("Saving the model...")
          start_time = time.time()
          # save the model
          model.saver.save(sess, os.path.normpath(os.path.join(train_dir, 'checkpoint')), global_step=current_step)

          # save log file
          with open(train_dir + "/train_log.txt", "a") as log_file:
            log_file.write("\nData Logger\n")
            log_file.write("--------------------------\n")
            log_file.write("{0: <16} |".format("data_millisec"))
            for ms in [80, 160, 320, 400, 560, 1000]:
              log_file.write(" {0:5d} |".format(ms))
            log_file.write("\n")
            log_file.write("{0: <16} |".format("error_expmap "))
            for ms in [1, 3, 7, 9, 13, 24]:
              if FLAGS.seq2seq_output_size >= ms + 1:
                log_file.write(" {0:.3f} |".format(test_step_euler_losses[ms]))
              else:
                log_file.write("   n/a |")
            log_file.write("\n--------------------------\n")
            log_file.write("Global step:         %d\n"
                  "Learning rate:       %.4f\n"
                  "Step-time (ms):     %.4f\n"
                  "Train step loss avg (expmap):      %.4f\n"
                  "--------------------------\n"
                  "Test step loss (expmap):            %.4f\n"
                  "Test step loss (euler angle):       %.4f\n"
                  % (model.global_step.eval(), model.learning_rate.eval(), step_time * 1000, step_avg_loss,
                     test_step_loss, test_step_euler_loss))
            log_file.write("--------------------------\n")

          print("done in {0:.2f} ms\n".format((time.time() - start_time) * 1000))

        # Reset global time and loss
        step_time, step_avg_loss = 0, 0
        sys.stdout.flush()

    # 이전에 사용된 적이 있는 경우, 시각화용 파일 삭제
    try:
      os.remove(visualize_dir)
    except OSError:
      pass

    for action in actions:
      # test data 중 하나의 배치를 가져와서 순전파 step 을 진행
      encoder_inputs, decoder_inputs, decoder_outputs = data_utils.get_seq2seq_batch(test_data,
            FLAGS.train_batch_size, FLAGS.seq2seq_input_size, FLAGS.seq2seq_output_size, FLAGS.seq2seq_train_datapoint_size)
      step_avg_loss, model_output, _ = model.step(sess, encoder_inputs, decoder_inputs, decoder_outputs, True)

      # 시각화를 위한 데이터 변환
      data_utils.save_visualize_data(visualize_dir, action, decoder_outputs, model_output, normalize_parameter)

      # 시각화 화면 출력
      forward_kinematics.visualize_data(visualize_dir, action)

t = True
if t:
  train()
else:
  for action in define_actions(FLAGS.action):
    forward_kinematics.visualize_data(visualize_dir, action)

