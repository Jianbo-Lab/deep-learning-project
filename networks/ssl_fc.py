import numpy as np
import tensorflow as tf
from ops import *

class Encoder1:
    def __init__(self, hidden_dim = 100):
        self.hidden_dim = hidden_dim

    def __call__(self, x, z_dim, y_dim, reuse=None):
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


        h0 = tf.nn.softplus(linear(x, self.hidden_dim, scope = 'en_fc0', reuse=reuse))

        h1 = tf.nn.softplus(linear(h0, self.hidden_dim, scope = 'en_fc1', reuse=reuse))

        z_log_sigma_sq = linear(h1, z_dim, scope = 'z_log_sigma_sq', reuse=reuse)

        y_prob = tf.nn.softmax(linear(h1, y_dim, scope = 'y_prob', reuse=reuse))

        #h2 = tf.nn.softplus(linear(tf.concat(1, (h1, labels)), self.hidden_dim, scope = 'en_fc2'))

        #z_log_sigma_sq = linear(h2, z_dim, scope = 'z_log_sigma')
        return (z_log_sigma_sq, y_prob, h1)

class Encoder2:
    def __init__(self, hidden_dim = 100):
        self.hidden_dim = hidden_dim

    def __call__(self, h1, y, z_dim, reuse=None):
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

        h2 = tf.nn.softplus(linear(tf.concat(1, (h1, y)), self.hidden_dim, scope = 'en_fc2', reuse=reuse))

        z_mu= linear(h2, z_dim, scope = 'z_mu', reuse=reuse)
        return z_mu

class Decoder:
    def __init__(self, hidden_dim = 100):
        self.hidden_dim = hidden_dim

    def __call__(self, z, y, img_dim, reuse=None):
        """
        The probabilistic decoder which computes the mean of x drawn from
        the Bernoulli distribution p(x|z).

        Inputs:
        z: A batch of hidden variables.
        img_dim: The dimension of one input image.

        Outputs:
        x_mean: A batch of the means of p(x|z), each corresponding to a single z.
        """

        h0 = tf.nn.softplus(linear(tf.concat(1, (z, y)), self.hidden_dim, scope = 'de_fc0', reuse=reuse))

        h1 = tf.nn.softplus(linear(h0, self.hidden_dim, scope = 'de_fc1', reuse=reuse))

        x_mean = tf.nn.sigmoid(linear(h1, img_dim, scope = 'x_mean', reuse=reuse))
        return x_mean


