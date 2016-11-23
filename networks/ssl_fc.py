import numpy as np
import tensorflow as tf
#from ops import *
slim=tf.contrib.slim

class SSL_Encoder1:
    def __init__(self, hidden_dim = 100):
        self.hidden_dim = hidden_dim

    def __call__(self, x, z_dim, y_dim, reuse=None, train_phase=True):
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


        self.hidden_dims =[self.hidden_dim]*2

        net = x
        for i in xrange(len(self.hidden_dims)):
            net = slim.fully_connected(net, self.hidden_dims[i],
                activation_fn=tf.nn.softplus,
                scope='enc_fc{}'.format(i),
                reuse=reuse,
                normalizer_fn=slim.batch_norm,
                normalizer_params={'reuse':reuse,
                    'is_training':train_phase,
                    'scale':True,
                    'updates_collections':None,
                    'scope':'enc_bn{}'.format(i)},
                )
            #net = slim.fully_connected(net, self.hidden_dims[i], reuse=reuse, scope='enc_fc{}'.format(i))
            #net = slim.batch_norm(net, activation_fn=tf.nn.softplus, reuse=reuse, is_training=train_phase, scope='enc_bn{}'.format(i), scale=True, updates_collections=None)
        h1 = net


        """
        h0 = tf.nn.softplus(batch_norm_layer(
            linear(x, self.hidden_dim, scope = 'en_fc0', reuse=reuse),
            train_phase=train_phase, scope_bn='en_bn0', reuse=reuse
            ))

        h1 = tf.nn.softplus(batch_norm_layer(
            linear(h0, self.hidden_dim, scope = 'en_fc1', reuse=reuse),
            train_phase=train_phase, scope_bn='en_bn1', reuse=reuse
            ))
        """

        z_log_sigma_sq = slim.fully_connected(h1, z_dim, scope='z_log_sigma_sq', reuse=reuse, activation_fn=None)


        y_prob = slim.fully_connected(h1, y_dim, scope='y_prob', reuse=reuse,
            activation_fn=tf.nn.softmax, normalizer_fn=slim.batch_norm,
            normalizer_params={'reuse':reuse,
                'is_training':train_phase,
                'scale':True,
                'updates_collections':None,
                'scope':'y_prob_bn'})
        #y_prob = slim.batch_norm(
            #slim.fully_connected(h1, y_dim, scope='y_prob', reuse=reuse),
            #is_training=train_phase, scope='y_prob_bn', reuse=reuse, activation_fn=tf.nn.softmax,
            #scale=True, updates_collections=None
            #)


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
    def __init__(self, hidden_dim = 100):
        self.hidden_dim = hidden_dim

    def __call__(self, z, y, img_dim, reuse=None, train_phase=True):
        """
        The probabilistic decoder which computes the mean of x drawn from
        the Bernoulli distribution p(x|z).

        Inputs:
        z: A batch of hidden variables.
        y: labels
        img_dim: The dimension of one input image.

        Outputs:
        x_mean: A batch of the means of p(x|y,z)
        """



        self.hidden_dims = [self.hidden_dim]*2


        net = tf.concat(1, (z,y))
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
        #x_mean = slim.batch_norm(
            #slim.fully_connected(net, img_dim, reuse=reuse, scope='x_mean',
            #activation_fn=tf.nn.sigmoid),
            #scope='x_mean_bn', reuse=reuse, is_training=train_phase, activation_fn=tf.nn.sigmoid,
            #scale=True, updates_collections=None
            #)

        """
        h0 = tf.nn.softplus(batch_norm_layer(
            linear(tf.concat(1, (z, y)), self.hidden_dim, scope = 'de_fc0', reuse=reuse),
            train_phase=train_phase, scope_bn='de_bn0', reuse=reuse
            ))

        h1 = tf.nn.softplus(batch_norm_layer(
            linear(h0, self.hidden_dim, scope = 'de_fc1', reuse=reuse),
            train_phase=train_phase, scope_bn='de_bn1', reuse=reuse
            ))
        x_mean = tf.nn.sigmoid(batch_norm_layer(
            linear(h1, img_dim, scope = 'x_mean', reuse=reuse),
            train_phase=train_phase, scope_bn='x_mean_bn', reuse=reuse
            ))

        """

        return x_mean


