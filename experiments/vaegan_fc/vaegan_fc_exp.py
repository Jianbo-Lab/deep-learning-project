import sys
sys.path.append('../../core/')
sys.path.append('../../networks/')
from vaegan_fc import *
from vaegan import VAEGAN
from misc_ops import *

import numpy as np
import tensorflow as tf

#%matplotlib inline

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%load_ext autoreload
#%autoreload 2


from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from dataset import DataSet
mnist = read_data_sets('MNIST_data', one_hot=True)

tf.reset_default_graph()

# Train
sess = tf.InteractiveSession()
build_encoder = Encoder(512)
build_decoder = Decoder(512)
build_discriminator = Discriminator(1024)


if len(sys.argv) == 1:
    model = VAEGAN(sess, build_encoder, build_decoder, build_discriminator, dataset=mnist.train,
            batch_size = 100, z_dim = 50, x_dim = 784,
           learning_rate = 1e-3, num_epochs = 10, load=False, lr_decay=1.,
          checkpoint_name='vaegan_fc_checkpoint', checkpoint_dir = './'
    )
else:
    model = VAEGAN(sess, build_encoder, build_decoder, build_discriminator, dataset=mnist.train,
            batch_size = 100, z_dim = 50, x_dim = 784,
           learning_rate = 1e-3, num_epochs = 10, load=False, lr_decay=1.,
               load_file = sys.argv[1],
               checkpoint_name='vaegan_fc_checkpoint_loaded', checkpoint_dir = './'
    )

model.train()



