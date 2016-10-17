import numpy as np
import tensorflow as tf
from ops import *
hidden_dim = 100

def build_encoder(images, z_dim):
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

    h0 = tf.nn.softplus(linear(images, hidden_dim, scope = 'en_fc0'))

    h1 = tf.nn.softplus(linear(h0, hidden_dim, scope = 'en_fc1'))

    z_mean = linear(h1, z_dim, scope = 'z_mean')
    z_log_sigma_sq = linear(h1, z_dim, scope = 'z_log_sigma')
    return (z_mean, z_log_sigma_sq)


def build_decoder(z, img_dim):
    """
    The probabilistic decoder which computes the mean of x drawn from
    the Bernoulli distribution p(x|z).

    Inputs:
    z: A batch of hidden variables.
    img_dim: The dimension of one input image.

    Outputs:
    x_mean: A batch of the means of p(x|z), each corresponding to a single z.
    """

    h0 = tf.nn.softplus(linear(z, hidden_dim, scope = 'de_fc0'))

    h1 = tf.nn.softplus(linear(h0, hidden_dim, scope = 'de_fc1'))

    x_mean = tf.nn.sigmoid(linear(h1, img_dim, scope = 'x_mean'))
    return x_mean


