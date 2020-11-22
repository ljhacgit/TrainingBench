"""
Partially use code from,
Julieta Martinez, Javier Romero ( MIT License Copyright (c) 2016 )
"""

import tensorflow as tf
import numpy as np
import h5py

import os
import copy
# pylint: disable=redefined-builtin
from six.moves import xrange


# -------------------------------------------------------------------------------------------------------------------
# Rotation Matrix Transformation
# -------------------------------------------------------------------------------------------------------------------
def rotmat2euler(R):
    """
    Converts a rotation matrix to Euler angles
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1

    Args
      R: a 3x3 rotation matrix
    Returns
      eul: a 3x1 Euler angle representation of R
    """

    if R[0, 2] == 1 or R[0, 2] == -1:
        # special case
        E3 = 0  # set arbitrarily
        dlta = np.arctan2(R[0, 1], R[0, 2])

        if R[0, 2] == -1:
            E2 = np.pi / 2
            E1 = E3 + dlta
        else:
            E2 = -np.pi / 2
            E1 = -E3 + dlta

    else:
        E2 = -np.arcsin(R[0, 2])
        E1 = np.arctan2(R[1, 2] / np.cos(E2), R[2, 2] / np.cos(E2))
        E3 = np.arctan2(R[0, 1] / np.cos(E2), R[0, 0] / np.cos(E2))

    eul = np.array([E1, E2, E3])
    return eul


def quat2expmap(q):
    """
    Converts a quaternion to an exponential map
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1

    Args
      q: 1x4 quaternion
    Returns
      r: 1x3 exponential map
    Raises
      ValueError if the l2 norm of the quaternion is not close to 1
    """

    if (np.abs(np.linalg.norm(q) - 1) > 1e-3):
        raise (ValueError, "quat2expmap: input quaternion is not norm 1")

    sinhalftheta = np.linalg.norm(q[1:])
    coshalftheta = q[0]

    r0 = np.divide(q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps))
    theta = 2 * np.arctan2(sinhalftheta, coshalftheta)
    theta = np.mod(theta + 2 * np.pi, 2 * np.pi)

    if theta > np.pi:
        theta = 2 * np.pi - theta
        r0 = -r0

    r = r0 * theta
    return r


def rotmat2quat(R):
    """
    Converts a rotation matrix to a quaternion
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4

    Args
      R: 3x3 rotation matrix
    Returns
      q: 1x4 quaternion
    """

    rotdiff = R - R.T

    r = np.zeros(3)
    r[0] = -rotdiff[1, 2]
    r[1] = rotdiff[0, 2]
    r[2] = -rotdiff[0, 1]
    sintheta = np.linalg.norm(r) / 2
    r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps)

    costheta = (np.trace(R) - 1) / 2

    theta = np.arctan2(sintheta, costheta)

    q = np.zeros(4)
    q[0] = np.cos(theta / 2)
    q[1:] = r0 * np.sin(theta / 2)
    return q


def rotmat2expmap(R):
    return quat2expmap(rotmat2quat(R))


def expmap2rotmat(r):
    """
    Converts an exponential map angle to a rotation matrix
    Matlab port to python for evaluation purposes
    I believe this is also called Rodrigues' formula
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m

    Args
      r: 1x3 exponential map
    Returns
      R: 3x3 rotation matrix
    """

    theta = np.linalg.norm(r)
    r0 = np.divide(r, theta + np.finfo(np.float32).eps)
    r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3, 3)
    r0x = r0x - r0x.T
    R = np.eye(3, 3) + np.sin(theta) * r0x + (1 - np.cos(theta)) * (r0x).dot(r0x)
    return R


# -------------------------------------------------------------------------------------------------------------------
# Import H36M Dataset, Dataset Normalize
# -------------------------------------------------------------------------------------------------------------------
def read_csv_as_float(filename):
    """
    Borrowed from SRNN code. Reads a csv and returns a float matrix.
    https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34

    Args
      filename: string. Path to the csv file
    Returns
      returnArray: the read data in a float32 matrix
    """
    returnArray = []
    lines = open(filename).readlines()
    for line in lines:
        line = line.strip().split(',')
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]))

    returnArray = np.array(returnArray)
    return returnArray


def get_raw_data(path_to_dataset, subjects, actions):
    """
    Borrowed from SRNN code. This is how the SRNN code reads the provided .txt files
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L270

    Args
      path_to_dataset: string. directory where the data resides
      subjects: list of numbers. The subjects to load
      actions: list of string. The actions to load
    Returns
      train_data: dictionary with {k:v}
        k=(subject, action, subaction), v=(nxd) un-normalized data
      }
      complete_data: nxd matrix with all the data. Used to normlization stats
    """

    labeled_data = {}
    complete_data = []

    for subj in subjects:
        for action_idx in np.arange(len(actions)):
            action = actions[action_idx]

            for subact in [1, 2]:  # subactions
                print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))

                filename = '{0}/S{1}/{2}_{3}.txt'.format(path_to_dataset, subj, action, subact)
                action_sequence = read_csv_as_float(filename)

                labeled_data[(subj, action, subact)] = action_sequence
                complete_data = copy.deepcopy(action_sequence)

    return labeled_data, complete_data


def normalization_parameter(complete_data):
    """"
    Also borrowed for SRNN code. Computes mean, stdev and dimensions to ignore.
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L33

    Args
      completeData: nx99 matrix with data to normalize
    Returns
      data_mean: vector of mean used to normalize the data
      data_std: vector of standard deviation used to normalize the data
      dimensions_to_ignore: vector with dimensions not used by the model
      dimensions_to_use: vector with dimensions used by the model
    """
    data_mean = np.mean(complete_data, axis=0)
    data_std = np.std(complete_data, axis=0)

    dimensions_to_ignore = []
    dimensions_to_use = []

    dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
    dimensions_to_use.extend(list(np.where(data_std >= 1e-4)[0]))

    data_std[dimensions_to_ignore] = 1.0

    return {"data_mean": data_mean, "data_std": data_std,
            "dim_to_ignore": dimensions_to_ignore, "dim_to_use": dimensions_to_use}


def normalize_data(data, normalize_parameter):
    """
    Normalize input data by removing unused dimensions, subtracting the mean and
    dividing by the standard deviation

    Args
      data: nx99 matrix with data to normalize
      data_mean: vector of mean used to normalize the data
      data_std: vector of standard deviation used to normalize the data
      dim_to_use: vector with dimensions used by the model
      actions: list of strings with the encoded actions
    Returns
      data_out: the passed data matrix, but normalized
    """
    data_mean = normalize_parameter["data_mean"]
    data_std = normalize_parameter["data_std"]
    dim_to_use = normalize_parameter["dim_to_use"]

    data_out = {}

    for key in data.keys():
        data_out[key] = np.divide((data[key] - data_mean), data_std)
        data_out[key] = data_out[key][:, dim_to_use]

    return data_out


def denormalize_data(normalized_data, normalize_parameter):
    """Borrowed from SRNN code. Reads a csv file and returns a float32 matrix.
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12

    Args
      normalizedData: nxd matrix with normalized data
      data_mean: vector of mean used to normalize the data
      data_std: vector of standard deviation used to normalize the data
      dimensions_to_ignore: vector with dimensions not used by the model
    Returns
      origData: data originally used to
    """
    data_mean = normalize_parameter["data_mean"]
    data_std = normalize_parameter["data_std"]
    dimensions_to_ignore = normalize_parameter["dim_to_ignore"]

    T = normalized_data.shape[0]
    D = data_mean.shape[0]

    origData = np.zeros((T, D), dtype=np.float32)
    dimensions_to_use = []
    for i in range(D):
        if i in dimensions_to_ignore:
            continue
        dimensions_to_use.append(i)
    dimensions_to_use = np.array(dimensions_to_use)

    origData[:, dimensions_to_use] = normalized_data

    # potentially ineficient, but only done once per experiment
    stdMat = data_std.reshape((1, D))
    stdMat = np.repeat(stdMat, T, axis=0)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.repeat(meanMat, T, axis=0)
    origData = np.multiply(origData, stdMat) + meanMat
    return origData


# -------------------------------------------------------------------------------------------------------------------
# Data Utils for Seq2Seq Model and Training
# -------------------------------------------------------------------------------------------------------------------
def get_preprocessed_data(actions, seq2seq_input_size, seq2seq_output_size, data_dir):
    print("Reading training data (seq_len_in: {0}, seq_len_out {1}).".format(
        seq2seq_input_size, seq2seq_output_size))

    # 학습용, 테스트용 배우 선별
    train_subject_ids = [1, 5, 6, 7, 8, 9, 11]

    # load_data 사용해 각각 데이터 불러옴
    # train_data : {(subject, action, subaction):[n, d]}, complete_train_data : [n, d] (data from subact 2)
    train_data, complete_train_data = get_raw_data(data_dir, train_subject_ids, actions)

    # 정규화 진행
    # Normalize, subtract mean, divide by stdev
    # normalize_parameter : {parameter_name : data}
    normalize_parameter = normalization_parameter(complete_train_data)
    # train_data, test_data : {(subject, action, subaction):[n, d]}
    train_data = normalize_data(train_data, normalize_parameter)
    print("done reading data.")

    # 반환
    return train_data, normalize_parameter


def get_dataset(data, batch_size, iterations, source_seq_len, target_seq_len, datapoint_size):
    # train_data, test_data : {(subject, action, subaction):[n, d]}

    # Get a random batch of data from the specified bucket, prepare for step.
    # get all dic id
    all_keys = list(data.keys())

    # select dataset size
    dataset_size = max(batch_size * iterations, 10000000)

    # 0~n 까지 수 중, batch_size 크기의 데이터를 선택
    # chosen_keys : [batch_size]
    chosen_keys = np.random.choice(len(all_keys), dataset_size)

    # How many frames in total do we need?
    total_frames = source_seq_len + target_seq_len

    enc_input = np.zeros((dataset_size, source_seq_len - 1, datapoint_size), dtype=float)
    dec_input = np.zeros((dataset_size, target_seq_len, datapoint_size), dtype=float)
    target = np.zeros((dataset_size, target_seq_len, datapoint_size), dtype=float)

    for i in xrange(dataset_size):
        # 랜덤으로 선택된 데이터 번호의 key(id)
        the_key = all_keys[chosen_keys[i]]
        # Get the number of frames, 행 길이가 프래임
        n, _ = data[the_key].shape

        # Sample somewhere in the middle
        idx = np.random.randint(16, n - total_frames)
        # Select the data around the sampled points
        # 특정 프래임부터 랜덤으로 설정해 잘라냄
        data_sel = data[the_key][idx:idx + total_frames, :]

        # Add the data
        # 데이터를 추가
        enc_input[i, :, 0:datapoint_size] = data_sel[0:source_seq_len - 1, :]
        dec_input[i, :, 0:datapoint_size] = data_sel[source_seq_len - 1:source_seq_len + target_seq_len - 1, :]
        target[i, :, 0:datapoint_size] = data_sel[source_seq_len:, 0:datapoint_size]

    # [batch, seq_length, datapoint_size]
    return enc_input, dec_input, target


def get_seq2seq_batch(data, batch_size, source_seq_len, target_seq_len, datapoint_size):
    # train_data, test_data : {(subject, action, subaction):[n, d]}

    # Get a random batch of data from the specified bucket, prepare for step.
    # get all dic id
    all_keys = list(data.keys())
    # 0~n 까지 수 중, batch_size 크기의 데이터를 선택
    # chosen_keys : [batch_size]
    chosen_keys = np.random.choice(len(all_keys), batch_size)

    # How many frames in total do we need?
    total_frames = source_seq_len + target_seq_len

    enc_input = np.zeros((batch_size, source_seq_len - 1, datapoint_size), dtype=float)
    dec_input = np.zeros((batch_size, target_seq_len, datapoint_size), dtype=float)
    target = np.zeros((batch_size, target_seq_len, datapoint_size), dtype=float)

    for i in xrange(batch_size):
        # 랜덤으로 선택된 데이터 번호의 key(id)
        the_key = all_keys[chosen_keys[i]]
        # Get the number of frames, 행 길이가 프래임
        n, _ = data[the_key].shape

        # Sample somewhere in the middle
        idx = np.random.randint(16, n - total_frames)
        # Select the data around the sampled points
        # 특정 프래임부터 랜덤으로 설정해 잘라냄
        data_sel = data[the_key][idx:idx + total_frames, :]

        # Add the data
        # 데이터를 추가
        enc_input[i, :, 0:datapoint_size] = data_sel[0:source_seq_len - 1, :]
        dec_input[i, :, 0:datapoint_size] = data_sel[source_seq_len - 1:source_seq_len + target_seq_len - 1, :]
        target[i, :, 0:datapoint_size] = data_sel[source_seq_len:, 0:datapoint_size]

    # [batch, seq_length, datapoint_size]
    return enc_input, dec_input, target


def get_loss(step_output, target, normalize_parameter):
    # convert step_output, decoder_output to eular angle

    # batch_size = 0
    # convert step_output as [batch_size, seq_length, datapoint]

    batch_size, _, _ = step_output.shape
    # [batch_size, seq_length, datapoint]
    denorm_step_outputs = [denormalize_data(step_output[i, :, :], normalize_parameter) for i in range(batch_size)]
    denorm_target = [denormalize_data(target[i, :, :], normalize_parameter) for i in range(batch_size)]

    # mean_errors [batch_size, seq_length]
    batch_error = np.zeros((len(denorm_step_outputs), denorm_step_outputs[0].shape[0]))
    batch_euler_error = np.zeros((len(denorm_step_outputs), denorm_step_outputs[0].shape[0]))

    for i in range(len(denorm_step_outputs)):
        # get i th batch's data
        denorm_step_output = denorm_step_outputs[i]
        denorm_decoder_output = denorm_target[i]

        euc_error = np.power(denorm_decoder_output - denorm_step_output, 2)
        euc_error = np.sum(euc_error, 1)
        euc_error = np.sqrt(euc_error)
        batch_error[i, :] = euc_error

        # Convert from exponential map to Euler angles
        for j in np.arange(denorm_step_output.shape[0]):
            for k in np.arange(3, 97, 3):
                denorm_step_output[j, k:k + 3] = rotmat2euler(expmap2rotmat(denorm_step_output[j, k:k + 3]))
        for j in np.arange(denorm_decoder_output.shape[0]):
            for k in np.arange(3, 97, 3):
                denorm_decoder_output[j, k:k + 3] = rotmat2euler(expmap2rotmat(denorm_decoder_output[j, k:k + 3]))

        # The global translation (first 3 entries) and global rotation
        # (next 3 entries) are also not considered in the error, so the_key are set to zero.
        # See https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-249404882
        denorm_decoder_output[:, 0:6] = 0

        # Now compute the l2 error. The following is numpy port of the error
        # function provided by Ashesh Jain (in matlab), available at
        # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/motionGenerationError.m#L40-L54
        idx_to_use = np.where(np.std(denorm_decoder_output, 0) > 1e-4)[0]

        euc_error = np.power(denorm_decoder_output[:, idx_to_use] - denorm_step_output[:, idx_to_use], 2)
        euc_error = np.sum(euc_error, 1)
        euc_error = np.sqrt(euc_error)
        batch_euler_error[i, :] = euc_error

    return np.mean(batch_error, 0), np.mean(batch_euler_error, 0)


def get_checkpoint(train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir, latest_filename="checkpoint")
    print("train_dir", train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print("Loading model {0}".format(ckpt_name))
        return ckpt.model_checkpoint_path
    else:
        print("Could not find checkpoint. Aborting.")
        raise (ValueError, "Checkpoint {0} does not seem to exist".format(ckpt.model_checkpoint_path))


def save_visualize_data(visualize_dir, action, target, model_output, normalize_parameter):
    # 비정규화
    ground_truth = denormalize_data(target[0, :, :], normalize_parameter)
    # denormalizes too
    batch_size, _, _ = model_output.shape
    prediction_expmap = [denormalize_data(model_output[i, :, :], normalize_parameter) for i in range(batch_size)]

    # 샘플 파일을 저장함
    # Save the samples
    with h5py.File(visualize_dir, 'a') as hf:
        # Save conditioning ground truth
        node_name = 'expmap/gt/{0}'.format(action)
        hf.create_dataset(node_name, data=ground_truth)
        # Save prediction
        node_name = 'expmap/preds/{0}'.format(action)
        hf.create_dataset(node_name, data=prediction_expmap[0])
