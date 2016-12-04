import numpy as np
import tensorflow as tf
# from ops import *
slim = tf.contrib.slim

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



class Discriminator_conv:
    def  __init__(self, hidden_dim = 100):
        self.hidden_dim = hidden_dim
    def __call__(self, x, batch_size=100, reuse=None, x_width=28):

        #self.hidden_dims = [self.hidden_dim] * 4
        
        net = tf.reshape(x, [batch_size, x_width, x_width, -1])
        net = slim.conv2d(net, 20, [5,5], stride=1, reuse=reuse, scope='dis_conv0')
        net = slim.max_pool2d(net, [2,2], stride=2)
        net = slim.conv2d(net, 50, [5,5], stride=1, reuse=reuse, scope='dis_conv1')
        net = slim.max_pool2d(net, [2,2], stride=2)
        net = slim.conv2d(net, 100, [5,5], stride=1, reuse=reuse, scope='dis_conv2')
        net = tf.reshape(net, [batch_size, -1])
        net = slim.fully_connected(net, 500, reuse=reuse, scope='dis_fc0')
        #for i in xrange(len(self.hidden_dims)):
            #net = slim.fully_connected(net, self.hidden_dims[i], scope='dis_fc{}'.format(i), activation_fn=tf.tanh, reuse=reuse)
        p = slim.fully_connected(net, 1, scope='p', activation_fn=tf.nn.sigmoid, reuse=reuse)
        return p



class Discriminator_fc:
    def  __init__(self, hidden_dim = 100):
        self.hidden_dim = hidden_dim
    def __call__(self, x, reuse=None):

        self.hidden_dims = [self.hidden_dim] * 4
        net = x
        for i in xrange(len(self.hidden_dims)):
            net = slim.fully_connected(net, self.hidden_dims[i], scope='dis_fc{}'.format(i),
                activation_fn=tf.tanh, reuse=reuse)
        p = slim.fully_connected(net, 1, scope='p', activation_fn=tf.nn.sigmoid, reuse=reuse)
        return p



