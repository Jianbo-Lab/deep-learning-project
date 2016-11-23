import numpy as np
import tensorflow as tf
import cifar_read
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.models.image.cifar10 import cifar10_input
from tensorflow.models.image.cifar10 import cifar10
import time
import os

import tensorflow.contrib.slim as slim








class Variational_Autoencoder():
    def __init__(self, sess, build_encoder, build_decoder, checkpoint_name = 'deep_feat_vae_checkpoint',
        batch_size = 100, z_dim = 20, img_dim = 784, dataset = 'mnist',
                 load = False,load_file = None, learning_rate = 0.001,
                 loss_type = 'l2', checkpoint_dir = '../notebook/checkpoints/'):
        """
        Inputs:
        sess: TensorFlow session.
        build_encoder: A function that lays down the computational
        graph for the encoder.
        build_decoder: A function that lays down the computational
        graph for the decoder.
        checkpoint_name: The name of the checkpoint file to be saved.
        batch_size: The number of samples in each batch.
        z_dim: the dimension of z.
        img_dim: the dimension of an image.
        (Currently, we only consider 28*28 = 784.)
        dataset: The filename of the dataset.
        (Currently we only consider mnist.)
        learning_rate: The learning rate of the Adam optimizer.
        num_epochs: The number of epochs.

        """
        self.loss_type = loss_type

        self.sess = sess
        self.build_encoder = build_encoder
        self.build_decoder = build_decoder
        self.checkpoint_name = checkpoint_name
        self.z_dim = z_dim
        self.img_dim = img_dim
        self.batch_size = batch_size
        self.dataset = dataset

        self.load = load
        self.load_file = load_file
        self.checkpoint_dir = checkpoint_dir
        # if dataset == 'mnist':
        #     # Load MNIST data in a format suited for tensorflow.
        #     self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        #     self.n_samples = self.mnist.train.num_examples
        if dataset == 'mnist' and load == False:
            # Load MNIST data in a format suited for tensorflow.
            self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
            self.n_samples = self.mnist.train.num_examples
        ###########
        elif dataset == 'cifar10':
           
            self.cifar10_input = cifar_read.DataLoader()
            self.n_samples = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN


        else:
            self.n_samples = dataset.num_examples


        


        # Compute the objective function.
        self.loss = self.build_vae()

        self.learning_rate = learning_rate

        global_step = tf.Variable(0, trainable = False)
        self.optimum = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=global_step)


        # Build a graph that trains the model with one batch of examples
        # and updates the model parameters.
      
        # Build an initialization operation to run.
        init = tf.initialize_all_variables()
        self.sess.run(init)
        if self.load:
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, self.load_file)



    def train(self, num_epochs = 5, learning_rate = None):
        """ Train VAE for a number of steps."""

        self.num_epochs =  num_epochs
        if learning_rate != None:
            self.learning_rate = learning_rate
      

        num_batches = int(self.n_samples / self.batch_size)

        for epoch in xrange(self.num_epochs):
            avg_loss_value = 0.
            

            for b in xrange(num_batches):
                # Get images from the mnist dataset.
                batch_images = self.input()
                batch_images = batch_images.reshape(batch_images.shape[0],-1)

                # Sample a batch of eps from standard normal distribution.
                batch_eps = np.random.randn(self.batch_size, self.z_dim)

                # Run a step of adam optimization and loss computation.
                start_time = time.time()
                _, loss_value = self.sess.run([self.optimum,self.loss],
                                            feed_dict = {self.images: batch_images,
                                                        self.batch_eps: batch_eps})

                duration = time.time() - start_time

                

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                avg_loss_value += loss_value / self.n_samples * self.batch_size


            # Later we'll add summary and log files to record training procedures.
            # For now, we will satisfy with naive printing.
            print 'Epoch {} loss: {}'.format(epoch + 1, avg_loss_value)
        self.save(epoch)

    def save(self, epoch):
        self.saver = tf.train.Saver()
        self.saver.save(self.sess, os.path.join(self.checkpoint_dir,
            self.checkpoint_name))#, global_step = epoch)



    def de_loss(self, imgs1, imgs2, type):

        if type == 'l2':
            return(tf.reduce_sum((imgs1 - imgs2) ** 2))

        elif type == 'cross-entropy':
            return(-tf.reduce_sum(imgs1 * tf.log(1e-10 + imgs2) \
                           + (1 - imgs1) * tf.log(1e-10 + 1 - imgs2), 1))

        else:
            conv1_W_np = np.load('vgg_weight/conv1_W.npy')
            conv1_b_np = np.load('vgg_weight/conv1_b.npy')

            conv1_W = tf.constant(conv1_W_np, name="conv1_W")
            conv1_b = tf.constant(conv1_b_np, name="conv1_b")

            conv2_W_np = np.load('vgg_weight/conv2_W.npy')
            conv2_b_np = np.load('vgg_weight/conv2_b.npy')

            conv2_W = tf.constant(conv2_W_np, name="conv2_W")
            conv2_b = tf.constant(conv2_b_np, name="conv2_b")

            def conv2d_jc(x, W, b, strides=1):
                # Conv2D wrapper, with bias and relu activation
                x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
                x = tf.nn.bias_add(x, b)
                return tf.nn.relu(x)

            imgs1 = tf.reshape(imgs1 , [100, 30, 30, 3])
            imgs2 = tf.reshape(imgs2 , [100, 30, 30, 3])
            

            fm_11 = conv2d_jc(imgs1, conv1_W, conv1_b)
            fm_21 = conv2d_jc(imgs2, conv1_W, conv1_b)

            fm_12 = conv2d_jc(fm_11, conv2_W, conv2_b)
            fm_22 = conv2d_jc(fm_12, conv2_W, conv2_b)


            return(tf.reduce_sum((fm_11 - fm_21) ** 2) + tf.reduce_sum((fm_12 - fm_22) ** 2))
            
    def input(self):
        """
        This function reads in one batch of data.
        """
        if self.dataset == 'mnist':
            # Extract images and labels (currently useless) from the next batch.
            batch_images, _ = self.mnist.train.next_batch(self.batch_size)
            return batch_images

        elif self.dataset == 'cifar10':
            batch_images = self.cifar10_input.next_batch(batch_size=self.batch_size)
            return batch_images


        batch_images, _ = self.dataset.next_batch(self.batch_size)
        return(batch_images)

  

    def build_vae(self):
        """
        This function builds up VAE from encoder and decoder function
        in the global environment.
        """
        # Add a placeholder for one batch of images
        self.images = tf.placeholder(tf.float32,[self.batch_size,self.img_dim],
                                    name = 'images')

        # Create a placeholder for eps.
        self.batch_eps = tf.placeholder(tf.float32,[self.batch_size, self.z_dim], name = 'eps')


        # Construct the mean and the variance of q(z|x).
        self.encoder_mean, self.encoder_log_sigma2 = \
            self.build_encoder(self.images, self.z_dim)
        # Compute z from eps and z_mean, z_sigma2.
        self.batch_z = tf.add(self.encoder_mean, \
                        tf.mul(tf.sqrt(tf.exp(self.encoder_log_sigma2)), self.batch_eps))
        # Construct the mean of the Bernoulli distribution p(x|z).
        self.decoder_mean = self.build_decoder(self.batch_z,self.img_dim)


        # Compute the loss from decoder (empirically).
        decoder_loss = self.de_loss(self.images, self.decoder_mean, self.loss_type)
        # Compute the loss from encoder (analytically).
        encoder_loss = -0.5 * tf.reduce_sum(1 + self.encoder_log_sigma2
                                           - tf.square(self.encoder_mean)
                                           - tf.exp(self.encoder_log_sigma2), 1)

        # Add up to the cost.
        self.cost = tf.reduce_mean(encoder_loss + decoder_loss)

        return self.cost

    def generate(self,num = 10,load = False,filename = None,chengidea = False,cheng_perturb= 1e-7):
        """
        This function generates images from VAE.
        Input:
        num: The number of images we would like to generate from the VAE.
        """

        # At the current stage, we require the number of images to
        # be generated smaller than the batch size for convenience.
        assert num <= self.batch_size, \
        "We require the number of images to be generated smaller than the batch size."
        if chengidea:
            # Sample small perturbations.
            sampled_z = np.random.randn(self.batch_size,self.z_dim) * cheng_perturb
            # Sample one random vector and add to small perturbations.
            sampled_z += np.random.randn(self.z_dim)
        else:
            # Sample z from standard normals.
            sampled_z = np.random.randn(self.batch_size,self.z_dim)

        return self.sess.run(self.decoder_mean,\
                      feed_dict = {self.batch_z:sampled_z})

    def get_code(self, img):
        return self.sess.run(self.encoder_mean, feed_dict={self.images:img})


