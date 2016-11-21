import numpy as np
import tensorflow as tf
slim=tf.contrib.slim
from ops import *

class Encoder:
    def __init__(self, hidden_dims = [512,256]):
        self.hidden_dims = hidden_dims

    def __call__(self, x, N,K):
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
        #net = x
        #for i in xrange(len(self.hidden_dims)):
            #net = slim.fully_connected(net, self.hidden_dims[i])
            #net = slim.batch_norm(net)

        net = slim.stack(x, slim.fully_connected, self.hidden_dims)
        logits_z = tf.reshape(slim.fully_connected(net, N*K, activation_fn=None), [-1,K])
        q_z = tf.nn.softmax(logits_z)
        log_q_z = tf.log(q_z+1e-20)
        return log_q_z, logits_z

        #h0 = tf.nn.softplus(linear(x, self.hidden_dim, scope = 'en_fc0'))

        #h1 = tf.nn.softplus(linear(h0, self.hidden_dim, scope = 'en_fc1'))

        #z_mean = linear(h1, z_dim, scope = 'z_mean')
        #z_log_sigma_sq = linear(h1, z_dim, scope = 'z_log_sigma')
        #return (z_mean, z_log_sigma_sq)

class Decoder:
    def __init__(self, hidden_dims = [256,512]):
        self.hidden_dims = hidden_dims

    def __call__(self, z, x_dim):
        """
        The probabilistic decoder which computes the mean of x drawn from
        the Bernoulli distribution p(x|z).

        Inputs:
        z: A batch of hidden variables.
        img_dim: The dimension of one input image.

        Outputs:
        x_mean: A batch of the means of p(x|z), each corresponding to a single z.
        """

        #net = slim.flatten(z)
        #for i in xrange(len(self.hidden_dims)):
            #net = slim.fully_connected(net, self.hidden_dims[i])
            #net = slim.batch_norm(net)

        net = slim.stack(slim.flatten(z), slim.fully_connected, [256,512])
        out = slim.fully_connected(net, x_dim, activation_fn=None)
        return out

