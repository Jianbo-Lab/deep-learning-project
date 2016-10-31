import numpy as np
import tensorflow as tf
# from ops import *

class Encoder:
    def __init__(self, enc_size = 256):

        self.enc_size = enc_size

        self.lstm_enc = tf.nn.rnn_cell.LSTMCell(self.enc_size, state_is_tuple=True)

    def __call__(self, state, input, reuse):

        with tf.variable_scope("encoder",reuse = reuse):
                return self.lstm_enc(input, state)



class Decoder:

    def __init__(self, dec_size = 256):

        self.dec_size = dec_size

        self.lstm_dec = tf.nn.rnn_cell.LSTMCell(self.dec_size, state_is_tuple=True)

    def __call__(self, state, input, reuse):

        with tf.variable_scope("decoder",reuse = reuse):
                return self.lstm_dec(input, state)



