import sys
sys.path.append('../../core/')
sys.path.append('../../networks/')
from ssl_fc import Encoder1, Encoder2, Decoder
from m2 import SSL_M2
from misc_ops import *

import numpy as np
import tensorflow as tf
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
num_labeled = 5000
labeled, unlabeled = split_data(mnist.train, num_labeled)
_, y = labeled.next_batch(num_labeled)
# Test for adding summary.
# Train
sess = tf.InteractiveSession()
build_encoder1 = Encoder1(500)
build_encoder2 = Encoder2(500)
build_decoder = Decoder(500)
 
model = SSL_M2(sess, build_encoder1, build_encoder2, build_decoder, labeled, unlabeled,
           batch_size = 100, z_dim = 50, x_dim = 784, y_dim=10, alpha=55000./10,
          learning_rate = 1e-3, num_epochs = 1, load=False,
         checkpoint_name='SSL_M2_checkpoint_5000'
)
model.train()