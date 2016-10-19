import numpy as np
import tensorflow as tf
from ops import *
filter_dim = 32
hidden_dim = 20
class Encoder:
    def __init__(self,filter_dim = 32,hidden_dim = 20):
        self.filter_dim = filter_dim
        self.hidden_dim = hidden_dim

    def __call__(self,images,z_dim): 
        """
        The probabilistic encoder which computes the mean and the log 
        of variance of z drawn from the Gaussian distribution q(z|images). 
        The layers are:
        conv-BN-ReLU-conv-BN-ReLU-linear-BN-ReLU
        Finally we add a linear layer to convert to z_mean and z_log_sigma_sq.

        Both of the width and the height of the output from each 
        conv layer increases by twice those of the input, (if stride is set to 2.)
            
        Inputs:
        images: A batch of images. 
        z_dim: The dimension of hidden variable z.
        
        Outputs:
        A batch of means and log of the variances of z, each corresponding 
        to a single image in images.
        
        """
        bn0 = batch_norm(name='en_bn0')
        bn1 = batch_norm(name='en_bn1')
        bn2 = batch_norm(name='en_bn2')

        shape = images.get_shape().as_list()
        batch_size = shape[0]
        s = int(np.sqrt(shape[1]))

        images = tf.reshape(images, [batch_size, s, s, 1])

        h0 = tf.nn.relu(bn0(conv2d(images, self.filter_dim, name='en_conv0')))

        h1 = tf.nn.relu(bn1(conv2d(h0, self.filter_dim * 2, name='en_conv1')))

        h2 = tf.nn.relu(bn2(linear(tf.reshape(h1, [batch_size, -1]), self.hidden_dim, 'en_fc3')))

        z_mean = linear(h2,z_dim,scope = 'z_mean')

        z_log_sigma_sq = linear(h2,z_dim,scope = 'z_log_sigma')

        return (z_mean, z_log_sigma_sq)

class Decoder:
    def __init__(self,filter_dim = 32):
        self.filter_dim = filter_dim

    def __call__(self,z,img_dim):
        """
        The probabilistic decoder which computes the mean of x drawn from 
        the Bernoulli distribution p(x|z).
        The layers are:
        Linear-ReLU-deconv-BN-ReLU-deconv-BN-RelU

        Both of the width and the height of the output from each deconv 
        layer decreases by half of those of the input, (if stride is set to 2.)
        
        Inputs:
        z: A batch of hidden variables.
        img_dim: The dimension of one input image.
        
        Outputs:
        x_mean: A batch of the means of p(x|z), each corresponding to a single z.
        """
        shape = z.get_shape().as_list()
        batch_size = shape[0]

        bn0 = batch_norm(name='de_bn0')

        bn1 = batch_norm(name='de_bn1')

        s = int(np.sqrt(img_dim))
        s2, s4 = s, s#int(s/2), int(s/4) 


        h0 = linear(z, filter_dim * 2 * s4 * s4, 'de_fc0')
        h0 = tf.reshape(h0, [-1, s4, s4, self.filter_dim * 2])
        h0 = tf.nn.relu(bn0(h0))

        h1 = deconv2d(h0, [batch_size, s2, s2, self.filter_dim], 
            name='de_deconv1')
        h1 = tf.nn.relu(bn1(h1))

        h2 = deconv2d(h1,[batch_size, s, s, 1], name='de_deconv2')
        h2 = tf.nn.sigmoid(h2)
        x_mean = tf.reshape(h2, [-1, img_dim])

        return x_mean