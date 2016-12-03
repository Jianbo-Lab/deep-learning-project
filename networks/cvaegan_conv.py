import numpy as np
import tensorflow as tf
# from ops import *
slim = tf.contrib.slim

class Encoder:
    def __init__(self, hidden_dim = 100):
        self.hidden_dim = hidden_dim

    def __call__(self, x, y, z_dim, batch_size=100, x_width=28):
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

        #self.hidden_dims = [self.hidden_dim] * 2
        net = tf.reshape(x, [batch_size, x_width, x_width, -1])
        net = slim.conv2d(net, 20, [5,5], stride=1)
        net = slim.max_pool2d(net, [2,2], stride=2)
        net = slim.conv2d(net, 50, [5,5], stride=1)
        net = slim.max_pool2d(net, [2,2], stride=2)
        net = tf.reshape(net, [batch_size, -1])
        net = tf.concat(1, (net,y))
        net = slim.fully_connected(net, 500)


        #for i in xrange(len(self.hidden_dims)):
            #net = slim.fully_connected(net, self.hidden_dims[i], scope='enc_fc{}'.format(i), activation_fn=tf.nn.softplus)
        z_mean = slim.fully_connected(net, z_dim, scope='z_mean', activation_fn=None)
        z_log_sigma_sq = slim.fully_connected(net, z_dim, scope='z_log_sigma', activation_fn=None)

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

    def __call__(self, z, x_dim, batch_size=100, reuse=None):
        """
        The probabilistic decoder which computes the mean of x drawn from
        the Bernoulli distribution p(x|z).

        Inputs:
        z: A batch of hidden variables.
        x_dim: The dimension of one input image.

        Outputs:
        x_mean: A batch of the means of p(x|z), each corresponding to a single z.
        """

        #self.hidden_dims = [self.hidden_dim] * 2
        net = z
        net = slim.fully_connected(net, 500, reuse=reuse, scope='dec_fc0')
        net = slim.fully_connected(net, 7*7*50, reuse=reuse, scope='dec_fc1')
        net = tf.reshape(net, [batch_size, 7, 7, 50])
        net = tf.image.resize_nearest_neighbor(net, [14, 14])
        net = slim.conv2d_transpose(net, 20, [5,5], stride=1, reuse=reuse, scope='dec_conv0')
        net = tf.image.resize_nearest_neighbor(net, [28, 28])
        net = slim.conv2d_transpose(net, 1, [5,5], stride=1, reuse=reuse, scope='dec_conv1')
        net = tf.reshape(net, [batch_size, -1])
        net = slim.fully_connected(net, 512, reuse=reuse, scope='dec_fc2')
        #for i in xrange(len(self.hidden_dims)):
            #net = slim.fully_connected(net, self.hidden_dims[i], scope='dec_fc{}'.format(i),
                #activation_fn=tf.nn.softplus, reuse=reuse)

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



