import numpy as np
import tensorflow as tf
from gumbel_ops import *
from tensorflow.examples.tutorials.mnist import input_data
import time
import os
slim=tf.contrib.slim
Bernoulli = tf.contrib.distributions.Bernoulli




class Variational_Autoencoder():
    def __init__(self, sess, build_encoder, build_decoder, checkpoint_name = 'vae_checkpoint',
        batch_size = 100, K=10, N=30, x_dim = 784, dataset = 'mnist',
        learning_rate = 0.001, lr_decay=0.9, num_epochs = 5,load = False,load_file = None,
        checkpoint_dir = '../notebook/checkpoints/',
        tau0=1.0, anneal_rate=0.00003, min_temp=0.5):
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
        x_dim: the dimension of an image.
        (Currently, we only consider 28*28 = 784.)
        dataset: The filename of the dataset.
        (Currently we only consider mnist.)
        learning_rate: The learning rate of the Adam optimizer.
        num_epochs: The number of epochs.

        """
        self.sess = sess
        self.build_encoder = build_encoder
        self.build_decoder = build_decoder
        self.checkpoint_name = checkpoint_name
        self.x_dim = x_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.load = load
        self.load_file = load_file
        self.checkpoint_dir = checkpoint_dir
        self.K=K # number of classes
        self.N=N # number of categorical distributions
        self.tau0 = tau0
        self.lr_decay = lr_decay
        self.anneal_rate = anneal_rate
        self.min_temp = min_temp
        self.np_temp = tau0
        # if dataset == 'mnist':
        #     # Load MNIST data in a format suited for tensorflow.
        #     self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        #     self.n_samples = self.mnist.train.num_examples
        if dataset == 'mnist' and load == False:
            # Load MNIST data in a format suited for tensorflow.
            self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
            self.n_samples = self.mnist.train.num_examples
        else:
            self.n_samples = dataset.num_examples


        global_step = tf.Variable(0, trainable = False)


        # Compute the objective function.
        self.loss = self.build_vae()
        self.lr = tf.constant(self.learning_rate)


        # Build a graph that trains the model with one batch of examples
        # and updates the model parameters.
        self.optimum = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=global_step, var_list=slim.get_model_variables())

        # Build an initialization operation to run.
        init = tf.initialize_all_variables()
        self.sess.run(init)
        if self.load:
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, self.load_file)

        self.num_iterations = 0

        self.log = []

    def train(self):
        """ Train VAE for a number of steps."""



        num_batches = int(self.n_samples / self.batch_size)

        for epoch in xrange(self.num_epochs):
            avg_loss_value = 0.

            for b in xrange(num_batches):
                # Get images from the mnist dataset.
                x = self.input()

                # Run a step of adam optimization and loss computation.
                start_time = time.time()
                _, loss_value = self.sess.run([self.optimum,self.loss],
                                            feed_dict = {self.x: x,
                                                        self.tau:self.np_temp,
                                                        self.lr: self.learning_rate})
                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                avg_loss_value += loss_value / self.n_samples * self.batch_size

                self.num_iterations += 1

                if self.num_iterations % 100 == 1:
                    self.log.append([self.num_iterations, self.np_temp, loss_value / self.n_samples * self.batch_size])

                if self.num_iterations % 1000 == 1:
                    self.np_temp = np.maximum(self.tau0*np.exp(-self.anneal_rate*self.num_iterations), self.min_temp)
                    self.learning_rate *= self.lr_decay
            # Later we'll add summary and log files to record training procedures.
            # For now, we will satisfy with naive printing.
            print 'Epoch {} loss: {}'.format(epoch + 1, avg_loss_value)
            # print 'Temp: {}'.format(self.np_temp)
        self.save(epoch)

    def save(self,epoch):
        self.saver = tf.train.Saver()
        self.saver.save(self.sess, os.path.join(self.checkpoint_dir,
            self.checkpoint_name))#, global_step = epoch)

    def input(self):
        """
        This function reads in one batch of data.
        """
        if self.dataset == 'mnist':
            # Extract images and labels (currently useless) from the next batch.
            x, _ = self.mnist.train.next_batch(self.batch_size)

            return x

        x, _ = self.dataset.next_batch(self.batch_size)
        return(x)

    def build_vae(self):
        """
        This function builds up VAE from encoder and decoder function
        in the global environment.
        """
        # Add a placeholder for one batch of images
        self.x = tf.placeholder(tf.float32,[self.batch_size,self.x_dim], name = 'x')



        # Encoder mean
        self.log_q_z, self.logits_z = self.build_encoder(self.x, self.N, self.K)

        self.tau = tf.Variable(5.0,name="temperature")

        # sample and reshape to (batch_size, N, K)
        self.z = tf.reshape(gumbel_softmax(self.logits_z, self.tau, hard=False), [-1,self.N,self.K])

        self.logits_x = self.build_decoder(self.z, self.x_dim)
        p_x = Bernoulli(logits = self.logits_x)
        self.x_mean = p_x.p
        # self.p_x_prob = tf.nn.sigmoid(self.logits_x)
        # self.x_mean = self.build_decoder(self.z, self.x_dim)

        self.encoder_loss = tf.reduce_sum(tf.reshape(
            tf.exp(self.log_q_z) * (self.log_q_z - tf.log(1.0/self.K)),
            [-1,self.N, self.K]
            ), [1,2])

        self.decoder_loss = -tf.reduce_sum(p_x.log_prob(self.x), 1)
        # self.decoder_loss = -tf.reduce_sum(self.x * tf.log(self.x_mean + 1e-10) + (1-self.x) * tf.log(1 - self.x_mean + 1e-10), 1)

        # Add up to the cost.
        self.cost = tf.reduce_mean(self.encoder_loss + self.decoder_loss)

        return self.cost

    def generate(self,num = 10,load = False,filename = None):
        """
        This function generates images from VAE.
        Input:
        num: The number of images we would like to generate from the VAE.
        """

        # At the current stage, we require the number of images to
        # be generated smaller than the batch size for convenience.
        assert num <= self.batch_size
        # Sample z from standard normals.

        #sampled_z = tf.reshape(gumbel_softmax(self.logits_z, self.tau, hard=False), [-1,self.N,self.K])
        M = self.batch_size * self.N
        z = np.zeros((M, self.K))
        z[xrange(M), np.random.choice(self.K,M)] = 1
        z = np.reshape(z, [self.batch_size, self.N, self.K])


        #prob = self.sess.run(self.p_x_prob, feed_dict = {self.z:z})
        prob = self.sess.run(self.x_mean, feed_dict = {self.z:z})
        return prob

    def get_code(self, x):
        return self.sess.run(self.z, feed_dict={self.x:x})

