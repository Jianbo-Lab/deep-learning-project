import numpy as np
import tensorflow as tf
#from ops import *
slim=tf.contrib.slim

class SSL_Encoder1:
    def __init__(self, hidden_dim = 100):
        self.hidden_dim = hidden_dim

    def __call__(self, x, z_dim, y_dim, reuse=None, train_phase=True, x_width=28, batch_size=100):
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


        #self.hidden_dims =[self.hidden_dim]*2

        net = x

        net = tf.reshape(x, [batch_size, x_width, x_width, -1])
        net = slim.conv2d(net, 20, [5,5], stride=1, reuse=reuse,
            scope='enc_conv1',
            normalizer_fn=slim.batch_norm,
                normalizer_params={'reuse':reuse,
                    'is_training':train_phase,
                    'scale':True,
                    'updates_collections':None,
                    'scope':'enc_conv_bn1'},
                )
        net = slim.max_pool2d(net, [2,2], stride=2)
        net = slim.conv2d(net, 50, [5,5], stride=1, reuse=reuse,
            scope='enc_conv2',
            normalizer_fn=slim.batch_norm,
                normalizer_params={'reuse':reuse,
                    'is_training':train_phase,
                    'scale':True,
                    'updates_collections':None,
                    'scope':'enc_conv_bn2'},
                )
        net = slim.max_pool2d(net, [2,2], stride=2)
        h1 = tf.reshape(net, [batch_size, -1]) 
        z_log_sigma_sq = slim.fully_connected(h1, z_dim, scope='z_log_sigma_sq', reuse=reuse, activation_fn=None)


        y_prob = slim.fully_connected(h1, y_dim, scope='y_prob', reuse=reuse,
            activation_fn=tf.nn.softmax,
            normalizer_fn=None) 


        """

        z_log_sigma_sq = linear(h1, z_dim, scope = 'z_log_sigma_sq', reuse=reuse)
        y_prob = tf.nn.softmax(batch_norm_layer(
            linear(h1, y_dim, scope = 'y_prob', reuse=reuse),
            train_phase=train_phase, scope_bn='y_prob_bn', reuse=reuse
            ))

        """
        return (z_log_sigma_sq, y_prob, h1)

class SSL_Encoder2:
    def __init__(self, hidden_dim = 100):
        self.hidden_dim = hidden_dim

    def __call__(self, h1, y, z_dim, reuse=None, train_phase=True):
        """
        The probabilistic encoder which computes the mean of z
        drawn from the Gaussian distribution q(z|images).

        Inputs:
        h1: output of previous encoder
        z_dim: latent dimension

        Outputs:
        A batch of means of z

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

        z_mu = slim.fully_connected(h2, z_dim, scope='z_mu', reuse=reuse, activation_fn=None)

        """
        h2 = tf.nn.softplus(batch_norm_layer(
            linear(tf.concat(1, (h1, y)), self.hidden_dim, scope = 'en_fc2', reuse=reuse),
            train_phase=train_phase, scope_bn='en_bn2', reuse=reuse
            ))
        z_mu= linear(h2, z_dim, scope = 'z_mu', reuse=reuse)


        """
        return z_mu

class SSL_Decoder:
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
            net = slim.fully_connected(z, network_size * network_size *128, reuse=reuse, scope='dec_fc0',
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


        """
        for i in xrange(len(self.hidden_dims)):
            net = slim.fully_connected(net, self.hidden_dims[i], scope='dec_fc{}'.format(i),
                reuse=reuse,
                activation_fn=tf.nn.softplus,
                normalizer_fn=slim.batch_norm,
                normalizer_params={'reuse':reuse,
                    'is_training':train_phase,
                    'scale':True,
                    'updates_collections':None,
                    'scope':'dec_bn{}'.format(i)})
                """
            #net = slim.fully_connected(net, self.hidden_dims[i], scope='dec_fc{}'.format(i), reuse=reuse)
            #net = slim.batch_norm(net, scope='dec_bn{}'.format(i), reuse=reuse, is_training=train_phase, activation_fn=tf.nn.softplus, scale=True, updates_collections=None)

        x_mean = slim.fully_connected(net, img_dim, scope='x_mean',
            reuse=reuse,
            activation_fn=tf.nn.sigmoid,
            normalizer_fn=slim.batch_norm,
            normalizer_params={'reuse':reuse,
                'is_training':train_phase,
                'scale':True,
                'updates_collections':None,
                'scope':'x_mean_bn'}) 

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
        net = slim.conv2d(net, network_size * 2, [3,3], stride=1, reuse=reuse, scope='dis_conv1',
            normalizer_fn=slim.batch_norm,
            normalizer_params={ 'reuse':reuse,
                                'is_training':train_phase,
                                'scale':True,
                                'updates_collections':None,
                                'scope':'dis_conv_bn1'})
        #net = slim.max_pool2d(net, [2,2], stride=2)
        net = slim.conv2d(net, network_size * 4, [3,3], stride=1, reuse=reuse, scope='dis_conv2',
            normalizer_fn=slim.batch_norm,
            normalizer_params={ 'reuse':reuse,
                                'is_training':train_phase,
                                'scale':True,
                                'updates_collections':None,
                                'scope':'dis_conv_bn2'})
        net = slim.conv2d(net, network_size * 8, [3,3], stride=1, reuse=reuse, scope='dis_conv3',
            normalizer_fn=slim.batch_norm,
            normalizer_params={ 'reuse':reuse,
                                'is_training':train_phase,
                                'scale':True,
                                'updates_collections':None,
                                'scope':'dis_conv_bn3'})
        net = tf.reshape(net, [batch_size, -1]) 
        # Add an extra layer. 
        net = slim.fully_connected(net, 512, reuse=reuse, scope='dis_fc0')
        # may delete the activation function.
        lth_layer = slim.fully_connected(net, 500, reuse=reuse, scope='dis_fc1')
        #for i in xrange(len(self.hidden_dims)):
            #net = slim.fully_connected(net, self.hidden_dims[i], scope='dis_fc{}'.format(i), activation_fn=tf.tanh, reuse=reuse)
        logits = slim.fully_connected(lth_layer, 11, scope='logits', activation_fn=tf.nn.sigmoid, reuse=reuse)
        return logits, lth_layer

