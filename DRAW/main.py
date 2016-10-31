
# coding: utf-8

# In[185]:

import sys
sys.path.append('../core/')
sys.path.append('../networks/')
from draw import DRAW
from networks import *

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython



# In[186]:

tf.reset_default_graph()


# In[187]:

sess = tf.InteractiveSession()
build_encoder = Encoder(256)
build_decoder = Decoder(256)
train_iters = 2
print_iters = 100


# In[188]:

model =  DRAW(sess, build_encoder, build_decoder, read_attn = True, write_attn = True)


# In[190]:

model.train(train_itrs = train_iters, learning_rate = 1e-4)


# In[191]:

plt.plot(range(train_iters), model.Lxs, range(train_iters), model.Lzs)


# In[192]:

generated_images = model.generate()


# In[193]:

generated_images.shape


# In[194]:

T = 10
num_examples = 5

plt.figure(figsize=(20,20))
for t in xrange(T):
    for n in xrange(num_examples):
        plt.subplot(num_examples, T, T* n + t + 1)
        plt.imshow(generated_images[t,n, ].reshape(28, 28), cmap='gray_r')
        plt.xticks([])
        plt.yticks([])
plt.tight_layout()
plt.show()

