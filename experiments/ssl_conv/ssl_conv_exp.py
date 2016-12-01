import sys
sys.path.append('../../core/')
sys.path.append('../../networks/')
from m2 import *
from ssl_conv2 import *
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

def split_data(dataset, num_labeled):
    """
    Split dataset into two datasets
    """
    n = dataset.num_examples
    x, y = dataset.next_batch(n)
    x1 = x[xrange(num_labeled),:]
    y1 = y[xrange(num_labeled),:]
    x2 = x[xrange(num_labeled, n),:]
    y2 = y[xrange(num_labeled, n),:]
    d1 = DataSet(x1, y1, dtype=dtypes.float32, reshape=False)
    d2 = DataSet(x2, y2, dtype=dtypes.float32, reshape=False)
    return d1, d2

# Split dataset into labeled and unlabeled
num_labeled = int(sys.argv[1])
labeled, unlabeled = split_data(mnist.train, num_labeled)
_, y = labeled.next_batch(num_labeled)
print np.sum(y, axis=0)

print 'Loaded data'

tf.reset_default_graph()

# Train
sess = tf.InteractiveSession()
build_encoder1 = SSL_Encoder1()
build_encoder2 = SSL_Encoder2()
build_decoder = SSL_Decoder()

print 'Started training'


if len(sys.argv) == 2:
    model = SSL_M2(sess, build_encoder1, build_encoder2, build_decoder, labeled, unlabeled,
            batch_size = 100, z_dim = 50, x_dim = 784, y_dim=10, alpha=55000./10,
           learning_rate = 5e-4, num_epochs = 10, load=False, lr_decay=0.99, lr_decay_freq=1000,
          checkpoint_name='ssl_conv_checkpoint', checkpoint_dir = './'
    )
else:
    model = SSL_M2(sess, build_encoder1, build_encoder2, build_decoder, labeled, unlabeled,
            batch_size = 100, z_dim = 50, x_dim = 784, y_dim=10, alpha=55000./10,
           learning_rate = 5e-4, num_epochs = 10, load=True, lr_decay=0.99, lr_decay_freq=1000,
               load_file = sys.argv[2],
               checkpoint_name='ssl_conv_checkpoint_loaded', checkpoint_dir = './'
    )

model.train()



