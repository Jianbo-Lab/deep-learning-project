import numpy as np
import tensorflow as tf
from ops import *

class Encoder:
    def __init__(self, hidden_dim = 100):
        self.hidden_dim = hidden_dim

    def __call__(self, images, z_dim):
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

        h0 = tf.nn.softplus(linear(images, self.hidden_dim, scope = 'en_fc0'))

        h1 = tf.nn.softplus(linear(h0, self.hidden_dim, scope = 'en_fc1'))

        z_mean = linear(h1, z_dim, scope = 'z_mean')
        z_log_sigma_sq = linear(h1, z_dim, scope = 'z_log_sigma')
        return (z_mean, z_log_sigma_sq)

class Decoder:
    def __init__(self, hidden_dim = 100):
        self.hidden_dim = hidden_dim

    def __call__(self, z, w, b, img_dim, batch_size):
        """
        The probabilistic decoder which computes the mean of x drawn from
        the Bernoulli distribution p(x|z).

        Inputs:
        z: A batch of hidden variables.
        img_dim: The dimension of one input image.

        Outputs:
        x_mean: A batch of the means of p(x|z), each corresponding to a single z.
        """





        h0 = tf.nn.softplus(tf.matmul(z, w[0]) + b[0])
        h1 = tf.nn.softplus(tf.matmul(z, w[1]) + b[1])
        x_mean = tf.nn.sigmoid(tf.matmul(z, w[2]) + b[2])
        return x_mean


