
# coding: utf-8

# This notebook re-runs part of the experiments in the tutorial https://jmetzen.github.io/2015-11-27/vae.html

# In[1]:

from __future__ import division, print_function, absolute_import
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression


# In[2]:

import sys
sys.path.append('../core/')
sys.path.append('../networks/')
import cifar_read
from simple_fc import Encoder, Decoder
from vae123 import Variational_Autoencoder

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[3]:

tf.reset_default_graph()


# In[4]:

sess = tf.InteractiveSession()
build_encoder = Encoder(100)
build_decoder = Decoder(100)
model = Variational_Autoencoder(sess, build_encoder, build_decoder, batch_size = 100,
                                z_dim = 20,img_dim = 2700,dataset = 'cifar10',
                                loss_type = 'deep_feat')


# In[5]:

model.train(num_epochs = 200, learning_rate = 0.01)

