import sys
import scipy.io as sio
sys.path.append('../../core/')
sys.path.append('../../networks/')
#sys.path.append('../../notebook/')
#from ssl_fc import *
#from ssl_conv import *
#from m2 import SSL_M2
#from vaegan_fc_svhn import *
from cvaegan_conv import *
from cvaegan import CVAEGAN
#from vaegan import VAEGAN
#from misc_ops import *

import numpy as np
import tensorflow as tf


from dataset import DataSet
from tensorflow.python.framework import dtypes

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%load_ext autoreload
#%autoreload 2


def mat2dataset(filepath):
    dat = sio.loadmat(filepath)
    x = dat['X']
    y = dat['y']
    y[y==10] = 0
    y = np.array(np.arange(10)==y, dtype=int)
    x = x.reshape((32*32*3,-1)).T
    x = x.astype(float)/255.
    return DataSet(x,y,dtype=dtypes.float32, reshape=False)

svhn_train = mat2dataset('../../notebook/SVHN/train_32x32.mat')
svhn_test = mat2dataset('../../notebook/SVHN/test_32x32.mat')

print 'Loaded SVHN'

tf.reset_default_graph()

sess = tf.InteractiveSession()
build_encoder = Encoder()
build_decoder = Decoder()
build_discriminator = Discriminator()

if len(sys.argv) == 1:
    model = CVAEGAN(sess, build_encoder, build_decoder, build_discriminator,
               checkpoint_name='cvaegan_svhn_conv', dataset=svhn_train,
               learning_rate=5e-4, lr_decay=1., num_epochs=5, x_dim=32*32*3,z_dim=100, x_width=32,
              gamma=1., checkpoint_dir = './'
              )
else:
    model = CVAEGAN(sess, build_encoder, build_decoder, build_discriminator,
               checkpoint_name='cvaegan_svhn_conv_loaded', dataset=svhn_train,
               learning_rate=1e-3, lr_decay=1., num_epochs=5, x_dim=32*32*3, z_dim=100, x_width=32,
              gamma=1., load=True, load_file=sys.argv[1], checkpoint_dir = './'
              )

print 'Starting training'

model.train()
