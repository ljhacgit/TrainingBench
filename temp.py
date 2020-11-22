import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU

import matplotlib.pylab as plt

enc_input = 0
dec_input = 0
initial_state = 0

enc_output, enc_state = GRU(128, return_sequences=True, return_state=True)(enc_input, initial_state)
dec_output, dec_state = GRU(128, return_state=True)(dec_input, enc_state)




