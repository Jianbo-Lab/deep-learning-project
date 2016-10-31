import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import time
import os
from helper import *



class DRAW():

    def __init__(self, sess, build_encoder, build_decoder, read_attn = False, write_attn = False, T = 10, dec_size = 256, enc_size = 256):
        self.sess = sess

        self.build_encoder = build_encoder
        self.build_decoder = build_decoder

    
        self.dec_size = dec_size
        self.enc_size = enc_size

        self.DO_SHARE = None # workaround for variable_scope(reuse=True)
        self.data_dir = ""
        self.read_attn = read_attn
        self.write_attn = write_attn

        # Currently I just fix all the tuning parameters. Could be given as input arguments.
        ## MODEL PARAMETERS ##

        self.A, self.B = 28, 28 # image width,height
        self.img_size = self.B * self.A # the canvas size
        # self.enc_size = 256 # number of hidden units / output size in LSTM
        # self.dec_size = 256
        self.read_n = 5 # read glimpse grid width/height
        self.write_n = 5 # write glimpse grid width/height
        self.read_size = 2* self.read_n * self.read_n if self.read_attn else 2* self.img_size
        self.write_size = self.write_n * self.write_n if self.write_attn else self.img_size
        self.z_size=10 # QSampler output size
        #####
        self.T = T # MNIST generation sequence length

        self.eps=1e-8 # epsilon for numerical stability
        
        ## BUILD MODEL ## 
    

    def train(self, train_itrs = 1000, batch_size = 100, print_itrs = 1000, learning_rate = 1e-4, load = False):

        self.batch_size = batch_size # training minibatch size
        self.train_itrs = train_itrs
        self.print_itrs = print_itrs
        self.learning_rate= learning_rate # learning rate for optimizer

        #---------------------------------------------------------------
        #---------------------------------------------------------------
        #---------------------------------------------------------------
        # Build Graph

        Lx, Lz = self.build_DRAW()

        cost = Lx + Lz

        ## OPTIMIZER ## 
        
        optimizer=tf.train.AdamOptimizer(self.learning_rate, beta1 = 0.5)
        grads=optimizer.compute_gradients(cost)

        for i,(g,v) in enumerate(grads):
            if g is not None:
                grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
        train_op = optimizer.apply_gradients(grads)
                
        ## RUN TRAINING ## 

        data_directory = os.path.join(self.data_dir, "mnist")
        if not os.path.exists(data_directory):
	    os.makedirs(data_directory)

        # binarized (0-1) mnist data

        train_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).train 

        fetches=[]
        fetches.extend([Lx, Lz, train_op])
        self.Lxs=[0] * self.train_itrs
        self.Lzs=[0] * self.train_itrs
        
        sess = self.sess
        
        self.saver = tf.train.Saver() # saves variables learned during training
        tf.initialize_all_variables().run()

        if load:
            self.saver.restore(sess, "draw.ckpt") # to restore from model, uncomment this line
        
        for i in range(self.train_itrs):

            # batch_eps = np.random.randn(self.T, self.batch_size, self.z_size)
	    xtrain,_ = train_data.next_batch(self.batch_size) # xtrain is (batch_size x img_size)

            self.feed_dict = {self.x : xtrain}

	    results = sess.run(fetches, self.feed_dict)
	    self.Lxs[i], self.Lzs[i], _ = results

	    if i % self.print_itrs  == 0:
		print("iter=%d : Lx: %f Lz: %f" % (i, self.Lxs[i], self.Lzs[i]))



        self.saver.save(sess,'draw.ckpt')



    def inference(self):

        # This function input a set of images
        # Return the drawed images during training.
        # Future work: I am working on generating image without input image, but have some issues with placeholder. So currently I made this compromising.
        canvases = self.sess.run(self.cs, self.feed_dict) # generate some examples
        canvases = np.array(canvases) # T x batch x img_size
        return canvases

    def generate(self):

        feed_dict = {}

        for t in range(self.T):
            feed_dict[self.z[t]] = np.random.randn(self.batch_size, self.z_size)

        canvases = self.sess.run(self.cs, feed_dict)
        canvases = np.array(canvases) # T x batch x img_size
        return canvases



    def build_DRAW(self):

        self.z = {}
        # Determine if we want to use 'attention' during reading and writing
        read = read_attn if self.read_attn else read_no_attn
        write = write_attn if self.read_attn else write_no_attn
      
        self.x = tf.placeholder(tf.float32,shape=(self.batch_size, self.img_size)) # input (batch_size * img_size)


        # Initialize variables
        self.cs = [0] * self.T # sequence of canvases
        mus, logsigmas, sigmas=[0] * self.T,[0] * self.T,[0] * self.T # gaussian params generated by SampleQ. We will need these for computing loss.
        # Initial states
        h_dec_prev = tf.zeros((self.batch_size, self.dec_size))
        enc_state = self.build_encoder.lstm_enc.zero_state(self.batch_size, tf.float32)
        dec_state = self.build_decoder.lstm_dec.zero_state(self.batch_size, tf.float32)

        # Construct the unrolled computational graph
        for t in range(self.T):
            # c is the current plot
            c_prev = tf.zeros((self.batch_size, self.img_size)) if t==0 else self.cs[t-1]
            # Error image
            x_hat = self.x - tf.sigmoid(c_prev) 
            # r is the input the the current encoder
            r = read(self.x, x_hat, h_dec_prev, self.A, self.B, self.read_n, self.DO_SHARE, self.eps)
            # Update the hidden encode state and encode.
            h_enc, enc_state = self.build_encoder(enc_state, tf.concat(1,[r, h_dec_prev]), reuse = self.DO_SHARE)


            self.z[t], mus[t], logsigmas[t], sigmas[t]= sampleQ(h_enc, self.DO_SHARE, \
                                                                  self.batch_size, self.z_size)

            # print self.z
            h_dec, dec_state = self.build_decoder(dec_state, self.z[t],  reuse = self.DO_SHARE)

  
            self.cs[t] = c_prev + write(h_dec, self.DO_SHARE, self.write_n, self.A, self.B, self.eps, self.batch_size) # store results
            h_dec_prev = h_dec
            self.DO_SHARE = True # from now on, share variables

            ## Compute the loss

        def binary_crossentropy(t , o):
            return -(t * tf.log(o + self.eps) + (1.0 - t) * tf.log(1.0 - o + self.eps))

        # reconstruction term appears to have been collapsed down to a single scalar value (rather than one per item in minibatch)
        x_recons = tf.nn.sigmoid(self.cs[-1])

        # after computing binary cross entropy, sum across features then take the mean of those sums across minibatches
        Lx = tf.reduce_sum(binary_crossentropy(self.x, x_recons),1) # reconstruction loss
        Lx = tf.reduce_mean(Lx)

        kl_terms=[0] * self.T

        for t in range(self.T):
            mu2 = tf.square(mus[t])
            sigma2 = tf.square(sigmas[t])
            logsigma = logsigmas[t]
            kl_terms[t]=  0.5 * tf.reduce_sum(mu2 + sigma2 - 2 * logsigma, 1) - self.T * .5 # each kl term is (1xminibatch)
            
        KL = tf.add_n(kl_terms) # this is 1xminibatch, corresponding to summing kl_terms from 1:T
        Lz = tf.reduce_mean(KL) # average over minibatches

         

        return (Lx, Lz)

