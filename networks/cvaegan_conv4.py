import numpy as np
import tensorflow as tf
# from ops import *
slim = tf.contrib.slim

class Encoder:
    def __init__(self, hidden_dim = 100, x_width=28):
        self.hidden_dim = hidden_dim
        self.x_width = x_width

    def __call__(self, x, y, z_dim, batch_size=100,train_phase = True):
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

        net = tf.reshape(x, [batch_size, self.x_width, self.x_width, -1])
        net = slim.conv2d(net, 64, [5,5], stride=2,
            normalizer_fn=slim.batch_norm,
            normalizer_params={'is_training':train_phase,
                'scale':True,
                'updates_collections':None,
                'scope':'enc_bn0'}) 
        net = slim.conv2d(net, 128, [5,5], stride=2,
            normalizer_fn=slim.batch_norm,
            normalizer_params={'is_training':train_phase,
                'scale':True,
                'updates_collections':None,
                'scope':'enc_bn1'}) 
        net = slim.conv2d(net, 256, [5,5], stride=2,
            normalizer_fn=slim.batch_norm,
            normalizer_params={'is_training':train_phase,
                'scale':True,
                'updates_collections':None,
                'scope':'enc_bn2'})
        net = tf.reshape(net, [batch_size, -1])
        net = tf.concat(1, (net,y)) 
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
    def __init__(self, hidden_dim = 100,x_width = 28, layer = 4,x_depth = 1):
        self.hidden_dim = hidden_dim
        self.x_width = x_width
        self.layer = 4
        self.x_depth = x_depth
    def __call__(self, z, x_dim, batch_size=100, reuse=None,train_phase = True):
        """
        The probabilistic decoder which computes the mean of x drawn from
        the Bernoulli distribution p(x|z).

        Inputs:
        z: A batch of hidden variables.
        x_dim: The dimension of one input image.

        Outputs:
        x_mean: A batch of the means of p(x|z), each corresponding to a single z.
        """
  

        if self.layer == 4:
            network_size = self.x_width / 4

            net = z
            net = slim.fully_connected(net, network_size * network_size *128, reuse=reuse, scope='dec_fc0',
                normalizer_fn=slim.batch_norm,
                normalizer_params={'reuse':reuse,
                    'is_training':train_phase,
                    'scale':True,
                    'updates_collections':None,
                    'scope':'dec_bn0'})
            net = tf.reshape(net, [batch_size, network_size , network_size , 128])   
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
            x_mean = slim.conv2d_transpose(net, self.x_depth, [1,1], stride=1, reuse=reuse, scope='x_mean', activation_fn=tf.nn.sigmoid)
            x_mean = tf.reshape(x_mean,[batch_size,-1])   

            return x_mean
        elif self.layer == 5:
            network_size = self.x_width / 8
            net = z
            net = slim.fully_connected(net, self.x_width*self.x_width*4, reuse=reuse, scope='dec_fc0',
                normalizer_fn=slim.batch_norm,
                normalizer_params={'reuse':reuse,
                    'is_training':train_phase,
                    'scale':True,
                    'updates_collections':None,
                    'scope':'dec_bn0'})
            net = tf.reshape(net, [batch_size, network_size, network_size, 256])   
            net = slim.conv2d_transpose(net, 256, [5,5], stride=2, reuse=reuse, scope='dec_conv0',
                normalizer_fn=slim.batch_norm,
                normalizer_params={'reuse':reuse,
                    'is_training':train_phase,
                    'scale':True,
                    'updates_collections':None,
                    'scope':'dec_bn1'})  
            net = slim.conv2d_transpose(net, 128, [5,5], stride=2, reuse=reuse, scope='dec_conv1',
                normalizer_fn=slim.batch_norm,
                normalizer_params={'reuse':reuse,
                    'is_training':train_phase,
                    'scale':True,
                    'updates_collections':None,
                    'scope':'dec_bn2'}) 
            net = slim.conv2d_transpose(net, 32, [5,5], stride=2, reuse=reuse, scope='dec_conv2',
                normalizer_fn=slim.batch_norm,
                normalizer_params={'reuse':reuse,
                    'is_training':train_phase,
                    'scale':True,
                    'updates_collections':None,
                    'scope':'dec_bn3'}) 
            x_mean = slim.conv2d_transpose(net, self.x_depth,  [1,1], stride=1, reuse=reuse, scope='x_mean', activation_fn=tf.nn.sigmoid)
            x_mean = tf.reshape(x_mean,[batch_size,-1])   

            return x_mean           

class Discriminator:
    def  __init__(self, hidden_dim = 100,x_width = 28):
        self.hidden_dim = hidden_dim
        self.x_width = x_width
    def __call__(self, x, batch_size=100, reuse=None, train_phase = True):

        #self.hidden_dims = [self.hidden_dim] * 4
        net = tf.reshape(x, [batch_size, self.x_width, self.x_width, -1])
        # net = slim.conv2d(net, self.x_width, [3,3], stride=1, reuse=reuse, scope='dis_conv0',
        #     normalizer_fn=slim.batch_norm,
        #     normalizer_params={ 'reuse':reuse,
        #                         'is_training':train_phase,
        #                         'scale':True,
        #                         'updates_collections':None,
        #                         'scope':'dis_conv_bn0'})
        #net = slim.max_pool2d(net, [2,2], stride=2) 
        network_size = 8 #self.x_width / 4
        net = slim.conv2d(net, 32, [5,5], stride=2, reuse=reuse, scope='dis_conv1',
            normalizer_fn=slim.batch_norm,
            normalizer_params={ 'reuse':reuse,
                                'is_training':train_phase,
                                'scale':True,
                                'updates_collections':None,
                                'scope':'dis_conv_bn1'})
        #net = slim.max_pool2d(net, [2,2], stride=2)
        net = slim.conv2d(net, 128, [5,5], stride=2, reuse=reuse, scope='dis_conv2',
            normalizer_fn=slim.batch_norm,
            normalizer_params={ 'reuse':reuse,
                                'is_training':train_phase,
                                'scale':True,
                                'updates_collections':None,
                                'scope':'dis_conv_bn2'})
        net = slim.conv2d(net, 256, [5,5], stride=2, reuse=reuse, scope='dis_conv3',
            normalizer_fn=slim.batch_norm,
            normalizer_params={ 'reuse':reuse,
                                'is_training':train_phase,
                                'scale':True,
                                'updates_collections':None,
                                'scope':'dis_conv_bn3'})
        net = slim.conv2d(net, 256, [5,5], stride=2, reuse=reuse, scope='dis_conv4',
            normalizer_fn=slim.batch_norm,
            normalizer_params={ 'reuse':reuse,
                                'is_training':train_phase,
                                'scale':True,
                                'updates_collections':None,
                                'scope':'dis_conv_bn4'})
        net = tf.reshape(net, [batch_size, -1])  
        # may delete the activation function.
        lth_layer = slim.fully_connected(net, 1024, reuse=reuse, scope='dis_fc1', activation_fn=tf.nn.elu)
        #for i in xrange(len(self.hidden_dims)):
            #net = slim.fully_connected(net, self.hidden_dims[i], scope='dis_fc{}'.format(i), activation_fn=tf.tanh, reuse=reuse)
        logits = slim.fully_connected(lth_layer, 11, scope='logits', activation_fn=None, reuse=reuse)
        return logits, lth_layer
