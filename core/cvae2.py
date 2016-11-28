import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import os
from misc_ops import *

class Conditional_Variational_Autoencoder():
    def __init__(self, sess, build_encoder, build_decoder,
        dataset,
        checkpoint_name = 'cvae_checkpoint',
        batch_size = 100, z_dim = 20, x_dim = 784, #dataset = 'mnist',
        learning_rate = 0.001,lr_decay=0.95, lr_decay_freq=1000, num_epochs = 5,
        condition_on_label = True, get_cond_info = None,
        cond_info_dim = 10,
        load=False, load_file = None, checkpoint_dir = './checkpoints/',
        model = None, layer = 3):
        """
        Inputs:
        sess: TensorFlow session.
        build_encoder: A function that lays down the computational
        graph for the encoder.
        build_decoder: A function that lays down the computational
        graph for the decoder.
        batch_size: The number of samples in each batch.
        z_dim: the dimension of z.
        x_dim: the dimension of an image.
        (Currently, we only consider 28*28 = 784.)
        dataset: The filename of the dataset.
        (Currently we only consider mnist.)
        learning_rate: The learning rate of the Adam optimizer.
        num_epochs: The number of epochs.
        conditional: function that takes image and outputs conditional info
        condition_on_label: if true, uses label as conditional info
        """
        self.sess = sess
        self.build_encoder = build_encoder
        self.build_decoder = build_decoder
        self.checkpoint_name = checkpoint_name
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.img_height = int(np.sqrt(x_dim))
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.get_cond_info = get_cond_info
        self.condition_on_label = condition_on_label
        self.cond_info_dim = cond_info_dim
        self.load = load
        self.load_file = load_file
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.layer = layer
        self.lr_decay_freq = lr_decay_freq
        self.lr_decay=lr_decay


        #assert (not (get_cond_info is None)) or condition_on_label, \
        #"Need to specify conditional information"

        if condition_on_label:
            self.cond_info_dim = 10


        #if dataset == 'mnist' and load == False:
            # Load MNIST data in a format suited for tensorflow.
            #self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
            #self.n_samples = self.mnist.train.num_examples
        self.n_samples = dataset.num_examples


        if (not model is None):
            allimg, _ = dataset.next_batch(self.n_samples)
            self.mean_img = np.mean(allimg, axis=0)
            x, _ = dataset.next_batch(2)
            self.mean_img = np.mean(x, axis=0)
            feats = get_feats(x, self.mean_img, self.model, self.layer)
            self.x_dim = 28*28 + feats.shape[1]


        """ Build vae """
        global_step = tf.Variable(0, trainable = False)

        self.num_epoch = 0
        self.num_iter = 0
        self.log = []
        self.lr = tf.constant(self.learning_rate)


        # Compute the objective function.
        self.loss = self.build_vae()


        # Build a graph that trains the model with one batch of examples
        # and updates the model parameters.
        self.optimum = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=global_step)

        # Build an initialization operation to run.
        init = tf.initialize_all_variables()
        self.sess.run(init)

        if load:
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, self.load_file)

    def train(self):

        start_time = time.time()


        num_batches = int(self.n_samples / self.batch_size)

        for epoch in xrange(self.num_epochs):
            epoch_start = time.time()

            avg_loss_value = 0.

            for b in xrange(num_batches):
                # Get images from the mnist dataset.
                batch_images, batch_info = self.input()

                # Sample a batch of eps from standard normal distribution.
                batch_eps = np.random.randn(self.batch_size,self.z_dim)

                # Run a step of adam optimization and loss computation.
                start_time = time.time()

                _, loss_value = self.sess.run([self.optimum, self.loss],
                                    feed_dict = {self.x: batch_images,
                                                self.y: batch_info,
                                                self.eps: batch_eps})
                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                avg_loss_value += loss_value / self.n_samples * self.batch_size

                self.num_iter += 1

                if self.num_iter % 100 == 1:
                    self.log.append([self.num_iter, loss_value, self.learning_rate])
                if self.num_iter % self.lr_decay_freq == 1:
                    self.learning_rate *= self.lr_decay

            # Later we'll add summary and log files to record training procedures.
            # For now, we will satisfy with naive printing.
            self.num_epoch += 1
            epoch_end = time.time()
            print 'Epoch {} loss: {} (time: {} s)'.format(self.num_epoch, avg_loss_value, epoch_end-epoch_start)
            self.save(self.num_epoch)
        end_time = time.time()
        print '{} min'.format((end_time-start_time)/60)
        #saver = tf.train.Saver()
        #saver.save(self.sess, 'cvae_checkpoint', global_step = epoch)
        #self.save(epoch)

    def save(self,epoch):
        self.saver = tf.train.Saver()
        #self.saver.save(self.sess, os.path.join(self.checkpoint_dir, self.checkpoint_name), global_step = epoch)
        self.saver.save(self.sess, os.path.join(self.checkpoint_dir, self.checkpoint_name), global_step=epoch)


    def input(self):
        """
        This function reads in one batch of data.
        """
        batch_images, batch_labels = self.dataset.next_batch(self.batch_size)
        if self.condition_on_label:
            batch_info = batch_labels
        else:
            batch_info = self.get_cond_info(batch_images)

        if not (self.model is None):
            feats = get_feats(batch_images, self.mean_img, self.model, self.layer)
            batch_images = np.concatenate((batch_images, feats), axis=1)

        return batch_images, batch_info

        #if self.dataset == 'mnist':
            # Extract images and labels (currently useless) from the next batch.
            #batch_images, batch_labels = self.mnist.train.next_batch(self.batch_size)



            #if self.condition_on_label:
                #batch_info = batch_labels
            #else:
                #batch_info = self.get_cond_info(batch_images)

            #return batch_images, batch_info

    def build_vae(self):
        """
        This function builds up VAE from encoder and decoder function
        in the global environment.
        """
        # Add a placeholder for one batch of images
        self.x = tf.placeholder(tf.float32,[self.batch_size, self.x_dim], name = 'images')

        # placeholder for cond info in CVAE
        self.y = tf.placeholder(tf.float32,[self.batch_size, self.cond_info_dim], name='info')

        # Create a placeholder for eps.
        self.eps = tf.placeholder(tf.float32,[self.batch_size, self.z_dim], name = 'eps')


        # Construct the mean and the variance of q(z|x).

        # fc
        #self.encoder_mean, self.encoder_log_sigma2 = self.build_encoder(tf.concat(1, (self.x, self.y)), self.z_dim)
        # conv
        self.encoder_mean, self.encoder_log_sigma2 = self.build_encoder(self.x, self.y, self.z_dim)


        # Compute z from eps and z_mean, z_sigma2.
        self.z = tf.add(self.encoder_mean, tf.mul(tf.sqrt(tf.exp(self.encoder_log_sigma2)), self.eps))




        # Construct the mean of the Bernoulli distribution p(x|z).
        self.decoder_mean = self.build_decoder(tf.concat(1, (self.z, self.y)),
                self.x_dim)

        # Compute the loss from decoder (empirically).
        decoder_loss = -tf.reduce_sum(self.x * tf.log(1e-10 + self.decoder_mean) \
                           + (1 - self.x) * tf.log(1e-10 + 1 - self.decoder_mean),
                           1)
        # Compute the loss from encoder (analytically).
        encoder_loss = -0.5 * tf.reduce_sum(1 + self.encoder_log_sigma2
                                           - tf.square(self.encoder_mean)
                                           - tf.exp(self.encoder_log_sigma2), 1)

        # Add up to the cost.
        self.cost = tf.reduce_mean(encoder_loss + decoder_loss)

        return self.cost

    def generate(self, num = 10, info = None):
        """
        This function generates images from VAE.
        Input:
        num: The number of images we would like to generate from the VAE.
        """

        # At the current stage, we require the number of images to
        # be generated smaller than the batch size for convenience.
        assert num <= self.batch_size, \
        "We require the number of images to be generated smaller than the batch size."

        # Sample z from standard normals.
        sampled_z = np.random.randn(self.batch_size,self.z_dim)

        if info is None:
            _, info = self.input()
        return self.sess.run(self.decoder_mean,
            feed_dict = {self.z: sampled_z, self.y: info})


