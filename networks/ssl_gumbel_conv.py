import numpy as np
import tensorflow as tf
#from ops import *
slim=tf.contrib.slim

class SSL_Encoder1:
    def __init__(self):#, hidden_dims = [512,256]):
        pass
        #self.hidden_dims = hidden_dims

    def __call__(self, x, y_dim, batch_size, reuse=None, train_phase=True):
        """
        The probabilistic encoder which computes the log
        of variance of z drawn from the Gaussian distribution q(z|images)
        as well as the categorical distribution q(y|x).

        Inputs:
        images: A batch of images.
        z_dim: The dimension of hidden variable z.
        y_dim: Number of labels

        Outputs:
        log variance, categorical distribution, last layer activations

        """
        net = tf.reshape(x, [-1, 28,28,1])
        with slim.arg_scope([slim.conv2d], stride=2, reuse=reuse):
            net = slim.stack(net, slim.conv2d, [
                (32, [5,5]),
                (64, [5,5]),
                (128, [5,5])
                ],
                scope = 'enc_conv')
        net = tf.reshape(net, [batch_size, -1])
        y_logit = slim.fully_connected(net, y_dim, reuse=reuse, scope='enc_fc')
        return y_logit

class SSL_Encoder2:
    def __init__(self):
        pass

    def __call__(self, x, y, z_dim, batch_size, reuse=None, train_phase=True):
        """
        The probabilistic encoder which computes the mean of z
        drawn from the Gaussian distribution q(z|images).

        Inputs:
        h1: output of previous encoder
        z_dim: latent dimension

        Outputs:
        A batch of means of z

        """

        #net = tf.concat(1, (x,y))
        net = tf.reshape(x, [-1,28,28,1])
        with slim.arg_scope([slim.conv2d], stride=2, reuse=reuse):
            net = slim.stack(net, slim.conv2d, [
                (32, [5,5]),
                (64, [5,5]),
                (128, [5,5])
                ],
                scope = 'enc2_conv')
        net = tf.reshape(net, [batch_size, -1])
        net = tf.concat(1, (net,y))
        z_log_sigma2 = slim.fully_connected(net, z_dim, reuse=reuse, scope='z_log_sigma2')
        z_mu = slim.fully_connected(net, z_dim, reuse=reuse, scope='z_mu')
        return (z_mu, z_log_sigma2)


class SSL_Decoder:
    def __init__(self):
        pass

    def __call__(self, z, y, x_dim, batch_size, reuse=None, train_phase=True):
        """
        The probabilistic decoder which computes the mean of x drawn from
        the Bernoulli distribution p(x|z).

        Inputs:
        z: A batch of hidden variables.
        y: labels
        x_dim: The dimension of one input image.

        Outputs:
        x_mean: A batch of the means of p(x|y,z)
        """

        net = tf.concat(1, (z,y))
        net = slim.fully_connected(net, 64, reuse=reuse, scope='dec_fc')
        net = tf.reshape(net, [-1, 8, 8, 1])
        with slim.arg_scope([slim.conv2d_transpose], stride=2, reuse=reuse):
            net = slim.stack(net, slim.conv2d_transpose, [
                (128, [3,3]),
                (64, [3,3]),
                (32, [3,3]),
                (32, [3,3])
                ],
                scope = 'dec_conv')
        net = tf.reshape(net, [batch_size, -1])
        x_logit = slim.fully_connected(net, x_dim, reuse=reuse, scope='dec_fc2')
        return x_logit


