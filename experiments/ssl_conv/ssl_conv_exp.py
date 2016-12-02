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

def split_data_even(dataset, num_per_label):
    """
    Split dataset into two datasets
    """
    n = dataset.num_examples
    x = dataset.images
    y = dataset.labels
    idx = []
    for i in xrange(10):
        subset = np.random.choice([j for j in xrange(n) if y[j,i]==1], num_per_label, False)
        idx = np.concatenate((idx, subset))
    idx = idx.astype(int)
    idx = np.random.choice(idx, len(idx), False)
    idx = idx.tolist()
    comp = list(set(range(n)) - set(idx))
    x1=x[idx,:]
    y1=y[idx,:]
    x2=x[comp,:]
    y2=y[comp,:]
    d1 = DataSet(x1, y1, dtype=dtypes.float32, reshape=False)
    d2 = DataSet(x2, y2, dtype=dtypes.float32, reshape=False)
    return d1, d2

labeled, unlabeled = split_data_even(mnist.train,int(sys.argv[1]))
print labeled.num_examples, unlabeled.num_examples
y = labeled.labels
print np.sum(y, axis=0)

print 'Loaded data'

tf.reset_default_graph()

# Train
sess = tf.InteractiveSession()
build_encoder1 = SSL_Encoder1(1024)
build_encoder2 = SSL_Encoder2(1024)
build_decoder = SSL_Decoder(1024)

print 'Started training'


if len(sys.argv) == 2:
    model = SSL_M2(sess, build_encoder1, build_encoder2, build_decoder, labeled, unlabeled,
            batch_size = 100, z_dim = 100, x_dim = 784, y_dim=10, alpha=55000./10,
           learning_rate = 5e-4, num_epochs = 10, load=False, lr_decay=0.99, lr_decay_freq=1000,
          checkpoint_name='ssl_conv_checkpoint', checkpoint_dir = './'
    )
else:
    model = SSL_M2(sess, build_encoder1, build_encoder2, build_decoder, labeled, unlabeled,
            batch_size = 100, z_dim = 100, x_dim = 784, y_dim=10, alpha=55000./10,
           learning_rate = 5e-4, num_epochs = 10, load=True, lr_decay=0.99, lr_decay_freq=1000,
               load_file = sys.argv[2],
               checkpoint_name='ssl_conv_checkpoint_loaded', checkpoint_dir = './'
    )

model.train()


# Classify validation images
batch_size = 100
num_val = mnist.validation.num_examples
tot = 0
for t in xrange(num_val / batch_size):
    x_val, y_val = mnist.validation.next_batch(batch_size)
    y_pred = model.classify(x_val)
    y_val = np.argmax(y_val, axis=1)
    tot += np.sum(y_pred != y_val)
print "Error: {}".format(float(tot)/num_val)
#with open("SSL_errors/SSL_err_1000_conv.txt", "w") as text_file:
    #text_file.write("Validation error: {}\n".format(float(tot)/num_val))


# Classify test images
batch_size = 100
num_test = mnist.test.num_examples
tot = 0
for t in xrange(num_test / batch_size):
    x_test, y_test = mnist.test.next_batch(batch_size)
    y_pred = model.classify(x_test)
    y_test = np.argmax(y_test, axis=1)
    tot += np.sum(y_pred != y_test)
print "Error: {}".format(float(tot)/num_test)
#with open("SSL_errors/SSL_err_1000_conv.txt", "a") as text_file:
    #text_file.write("Test error: {}".format(float(tot)/num_test))



