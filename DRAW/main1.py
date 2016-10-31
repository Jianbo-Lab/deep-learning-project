
# coding: utf-8

# In[28]:

import sys
sys.path.append('../core/')
sys.path.append('../networks/')
from cdraw import DRAW
from networks import *

import numpy as np
import tensorflow as tf


# In[29]:

tf.reset_default_graph()


# In[30]:

sess = tf.InteractiveSession()
build_encoder = Encoder(256)
build_decoder = Decoder(256)
train_iters = 50000
print_iters = 1000


# In[31]:

model =  DRAW(sess, build_encoder, build_decoder, read_attn = True, write_attn = True)


# In[32]:

model.train(train_itrs = train_iters, learning_rate = 1e-4, load = False, save = True)



# In[35]:

generated_images = model.generate()


np.save('generated_cdraw.txt', generated_images)
