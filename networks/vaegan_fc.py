import numpy as np
import tensorflow as tf
# from ops import *
slim = tf.contrib.slim

class Encoder:
    def __init__(self, hidden_dim = 100):
        self.hidden_dim = hidden_dim

    def __call__(self, x, z_dim):
        """
        The probabilistic encoder which computes the mean and the log
        of variance of z drawn from the Gaussian distribution q(z|images).

        Inputs:
        images: A batch of images.
        z_dim: The dimension of hidden variable z.

        Outputs:
        A batch of means and log of the variances of z, each corresponding
        to a single image in images.

        """

        self.hidden_dims = [self.hidden_dim] * 2
        net = x
        for i in xrange(len(self.hidden_dims)):
            net = slim.fully_connected(net, self.hidden_dims[i], scope='enc_fc{}'.format(i),
                activation_fn=tf.nn.softplus)
        z_mean = slim.fully_connected(net, z_dim, scope='z_mean',
            activation_fn=None)
        z_log_sigma_sq = slim.fully_connected(net, z_dim, scope='z_log_sigma',
            activation_fn=None)

        """

        h0 = tf.nn.softplus(linear(x, self.hidden_dim, scope = 'en_fc0'))

        h1 = tf.nn.softplus(linear(h0, self.hidden_dim, scope = 'en_fc1'))

        z_mean = linear(h1, z_dim, scope = 'z_mean')
        z_log_sigma_sq = linear(h1, z_dim, scope = 'z_log_sigma')
        """
        return (z_mean, z_log_sigma_sq)

class Decoder:
    def __init__(self, hidden_dim = 100):
        self.hidden_dim = hidden_dim

    def __call__(self, z, x_dim, reuse=None):
        """
        The probabilistic decoder which computes the mean of x drawn from
        the Bernoulli distribution p(x|z).

        Inputs:
        z: A batch of hidden variables.
        x_dim: The dimension of one input image.

        Outputs:
        x_mean: A batch of the means of p(x|z), each corresponding to a single z.
        """

        self.hidden_dims = [self.hidden_dim] * 2
        net = z
        for i in xrange(len(self.hidden_dims)):
            net = slim.fully_connected(net, self.hidden_dims[i], scope='dec_fc{}'.format(i),
                activation_fn=tf.nn.softplus, reuse=reuse)

        x_mean = slim.fully_connected(net, x_dim, scope='x_mean', activation_fn=tf.nn.sigmoid, reuse=reuse)

        """
        h0 = tf.nn.softplus(linear(z, self.hidden_dim, scope = 'de_fc0'))

        h1 = tf.nn.softplus(linear(h0, self.hidden_dim, scope = 'de_fc1'))

        x_mean = tf.nn.sigmoid(linear(h1, x_dim, scope = 'x_mean'))
        """
        return x_mean

class Discriminator:
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



