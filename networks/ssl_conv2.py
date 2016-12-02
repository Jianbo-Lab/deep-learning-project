import numpy as np
import tensorflow as tf
# from ops import *
slim = tf.contrib.slim

class SSL_Encoder1:
    def __init__(self, hidden_dim = 100):
        self.hidden_dim = hidden_dim

    def __call__(self, x, z_dim, y_dim, reuse=None, batch_size=100, x_width=28,train_phase = True):
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
        net = slim.conv2d(net, 28, [5,5], stride=2, reuse=reuse, scope='enc_conv0',
            normalizer_fn=slim.batch_norm,
            normalizer_params={'reuse':reuse,'is_training':train_phase,
                'scale':True,
                'updates_collections':None,
                'scope':'enc_bn0'})
        net = slim.conv2d(net, 56, [5,5], stride=2, reuse=reuse,scope='enc_conv1',
            normalizer_fn=slim.batch_norm,
            normalizer_params={'reuse':reuse,'is_training':train_phase,
                'scale':True,
                'updates_collections':None,
                'scope':'enc_bn1'})
        net = slim.conv2d(net, 108, [5,5], stride=2, reuse=reuse, scope='enc_conv2',
            normalizer_fn=slim.batch_norm,
            normalizer_params={'reuse':reuse,'is_training':train_phase,
                'scale':True,
                'updates_collections':None,
                'scope':'enc_bn2'})
        net = tf.reshape(net, [batch_size, -1])
        h1 = net
        #net = tf.concat(1, (net,y))
        #z_mean = slim.fully_connected(net, z_dim, scope='z_mean', activation_fn=None)
        z_log_sigma_sq = slim.fully_connected(net, z_dim, scope='z_log_sigma', activation_fn=None, reuse=reuse,
            normalizer_fn=slim.batch_norm,
            normalizer_params={'reuse':reuse,'is_training':train_phase,
                'scale':True,
                'updates_collections':None,
                'scope':'z_lgo_sigm_sq_bn'})

        y_prob = slim.fully_connected(h1, y_dim, scope='y_prob',
            activation_fn=tf.nn.softmax, reuse=reuse,
            normalizer_fn=slim.batch_norm,
            normalizer_params={'reuse':reuse,'is_training':train_phase,
                'scale':True,
                'updates_collections':None,
                'scope':'y_prob_bn'})

        """

        h0 = tf.nn.softplus(linear(x, self.hidden_dim, scope = 'en_fc0'))

        h1 = tf.nn.softplus(linear(h0, self.hidden_dim, scope = 'en_fc1'))

        z_mean = linear(h1, z_dim, scope = 'z_mean')
        z_log_sigma_sq = linear(h1, z_dim, scope = 'z_log_sigma')
        """
        return (z_log_sigma_sq, y_prob, h1)

class SSL_Encoder2:
    def __init__(self, hidden_dim = 512):
        self.hidden_dim = hidden_dim

    def __call__(self, h1, y, z_dim, reuse=None, train_phase=True):
        """
        The probabilistic encoder which computes the mean of z
        drawn from the Gaussian distribution q(z|images).
        """

        h2 = slim.fully_connected(tf.concat(1, (h1,y)), self.hidden_dim, scope='enc2_fc',
            activation_fn=tf.nn.softplus,
            normalizer_fn=slim.batch_norm,
            reuse=reuse,
            normalizer_params={'reuse':reuse,
                'is_training':train_phase,
                'scale':True,
                'updates_collections':None,
                'scope':'enc2_bn'})



        #h2 = slim.batch_norm(
            #slim.fully_connected(tf.concat(1, (h1,y)), self.hidden_dim, scope='enc2_fc', reuse=reuse),
            #is_training=train_phase, scope='enc2_bn', reuse=reuse, activation_fn=tf.nn.softplus,
            #scale=True, updates_collections=None
            #)

        z_mu = slim.fully_connected(h2, z_dim, scope='z_mu', reuse=reuse, activation_fn=None,
            normalizer_fn=slim.batch_norm,
            normalizer_params={'reuse':reuse,'is_training':train_phase,
                'scale':True,
                'updates_collections':None,
                'scope':'z_mu_bn'})

        """
        h2 = tf.nn.softplus(batch_norm_layer(
            linear(tf.concat(1, (h1, y)), self.hidden_dim, scope = 'en_fc2', reuse=reuse),
            train_phase=train_phase, scope_bn='en_bn2', reuse=reuse
            ))
        z_mu= linear(h2, z_dim, scope = 'z_mu', reuse=reuse)


        """
        return z_mu

class SSL_Decoder:
    def __init__(self, hidden_dim = 512):
        self.hidden_dim = hidden_dim

    def __call__(self, z, y, x_dim, batch_size=100, reuse=None,train_phase = True):
        """
        The probabilistic decoder which computes the mean of x drawn from
        the Bernoulli distribution p(x|z).

        Inputs:
        z: A batch of hidden variables.
        x_dim: The dimension of one input image.

        Outputs:
        x_mean: A batch of the means of p(x|z), each corresponding to a single z.
        """

        # net = z
        # net = slim.fully_connected(net, 8*8*256, reuse=reuse, scope='dec_fc0',
        #     normalizer_fn=slim.batch_norm,
        #     normalizer_params={'reuse':reuse,
        #         'is_training':train_phase,
        #         'scale':True,
        #         'updates_collections':None,
        #         'scope':'dec_bn0'})
        # net = tf.reshape(net, [batch_size, 8, 8, 256])
        # net = slim.conv2d_transpose(net, 256, [5,5], stride=2, reuse=reuse, scope='dec_conv0',
        #     normalizer_fn=slim.batch_norm,
        #     normalizer_params={'reuse':reuse,
        #         'is_training':train_phase,
        #         'scale':True,
        #         'updates_collections':None,
        #         'scope':'dec_bn1'})
        # net = slim.conv2d_transpose(net, 128, [5,5], stride=2, reuse=reuse, scope='dec_conv1',
        #     normalizer_fn=slim.batch_norm,
        #     normalizer_params={'reuse':reuse,
        #         'is_training':train_phase,
        #         'scale':True,
        #         'updates_collections':None,
        #         'scope':'dec_bn2'})
        # net = slim.conv2d_transpose(net, 32, [5,5], stride=2, reuse=reuse, scope='dec_conv2',
        #     normalizer_fn=slim.batch_norm,
        #     normalizer_params={'reuse':reuse,
        #         'is_training':train_phase,
        #         'scale':True,
        #         'updates_collections':None,
        #         'scope':'dec_bn3'})
        # x_mean = slim.conv2d_transpose(net, 1, [1,1], stride=1, reuse=reuse, scope='x_mean', activation_fn=tf.nn.sigmoid)
        # x_mean = tf.reshape(x_mean,[batch_size,-1])
        # print x_mean.get_shape()

        net = z
        net = slim.fully_connected(net, 7*7*128, reuse=reuse, scope='dec_fc0',
            normalizer_fn=slim.batch_norm,
            normalizer_params={'reuse':reuse,
                'is_training':train_phase,
                'scale':True,
                'updates_collections':None,
                'scope':'dec_bn0'})
        net = tf.reshape(net, [batch_size, 7, 7, 128])
        net = slim.conv2d_transpose(net, 64, [5,5], stride=2, reuse=reuse, scope='dec_conv0',
            normalizer_fn=slim.batch_norm,
            normalizer_params={'reuse':reuse,
                'is_training':train_phase,
                'scale':True,
                'updates_collections':None,
                'scope':'dec_bn1'})
        net = slim.conv2d_transpose(net, 32, [5,5], stride=2, reuse=reuse, scope='dec_conv1',
            normalizer_fn=slim.batch_norm,
            normalizer_params={'reuse':reuse,
                'is_training':train_phase,
                'scale':True,
                'updates_collections':None,
                'scope':'dec_bn2'})
        net = tf.reshape(net, [batch_size,-1])
        net = tf.concat(1, (net,y))
        net = slim.fully_connected(net, self.hidden_dim, scope='dec_fc1', reuse=reuse,
            normalizer_fn=slim.batch_norm,
            normalizer_params={'reuse':reuse,
                'is_training':train_phase,
                'scale':True,
                'updates_collections':None,
                'scope':'dec_bn3'})
        x_mean = slim.fully_connected(net, x_dim, scope='x_mean', reuse=reuse, activation_fn=tf.nn.sigmoid,
            normalizer_fn=slim.batch_norm,
            normalizer_params={'reuse':reuse,
                'is_training':train_phase,
                'scale':True,
                'updates_collections':None,
                'scope':'dec_bn4'})
        #x_mean = slim.conv2d_transpose(net, 1, [1,1], stride=1, reuse=reuse, scope='x_mean', activation_fn=tf.nn.sigmoid)
        #x_mean = tf.reshape(x_mean,[batch_size,-1])

        return x_mean

"""
class Discriminator:
    def  __init__(self, hidden_dim = 100):
        self.hidden_dim = hidden_dim
    def __call__(self, x, batch_size=100, reuse=None, x_width=28,train_phase = True):

        #self.hidden_dims = [self.hidden_dim] * 4
        net = tf.reshape(x, [batch_size, x_width, x_width, -1])
        net = slim.conv2d(net, 16, [3,3], stride=1, reuse=reuse, scope='dis_conv0',
            normalizer_fn=slim.batch_norm,
            normalizer_params={ 'reuse':reuse,
                                'is_training':train_phase,
                                'scale':True,
                                'updates_collections':None,
                                'scope':'dis_conv_bn0'})
        #net = slim.max_pool2d(net, [2,2], stride=2)
        net = slim.conv2d(net, 32, [3,3], stride=1, reuse=reuse, scope='dis_conv1',
            normalizer_fn=slim.batch_norm,
            normalizer_params={ 'reuse':reuse,
                                'is_training':train_phase,
                                'scale':True,
                                'updates_collections':None,
                                'scope':'dis_conv_bn1'})
        #net = slim.max_pool2d(net, [2,2], stride=2)
        net = slim.conv2d(net, 64, [3,3], stride=1, reuse=reuse, scope='dis_conv2',
            normalizer_fn=slim.batch_norm,
            normalizer_params={ 'reuse':reuse,
                                'is_training':train_phase,
                                'scale':True,
                                'updates_collections':None,
                                'scope':'dis_conv_bn2'})
        net = tf.reshape(net, [batch_size, -1])
        net = slim.fully_connected(net, 512, reuse=reuse, scope='dis_fc0')
        lth_layer = slim.fully_connected(net, 500, reuse=reuse, scope='dis_fc1')
        #for i in xrange(len(self.hidden_dims)):
            #net = slim.fully_connected(net, self.hidden_dims[i], scope='dis_fc{}'.format(i), activation_fn=tf.tanh, reuse=reuse)
        logits = slim.fully_connected(lth_layer, 11, scope='logits', activation_fn=tf.nn.sigmoid, reuse=reuse)
        return logits, lth_layer
"""
