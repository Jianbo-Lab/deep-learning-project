import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import os



class CVAEGAN():
    def __init__(self, sess, build_encoder, build_decoder, build_discriminator, checkpoint_name = 'vae_checkpoint',
        batch_size = 100, z_dim = 20, x_dim = 784, dataset = 'mnist', cond_info_dim=10,
        learning_rate = 0.001, lr_decay=0.95, lr_decay_freq=1000, num_epochs = 5,load = False,load_file = None, gamma=1.,
        checkpoint_dir = '../notebook/checkpoints/'):
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
        self.build_discriminator = build_discriminator
        self.checkpoint_name = checkpoint_name
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.load = load
        self.load_file = load_file
        self.checkpoint_dir = checkpoint_dir
        self.lr_decay=lr_decay
        self.lr_decay_freq=lr_decay_freq
        self.cond_info_dim = cond_info_dim
        self.gamma = gamma # down-weight GAN loss
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


        self.num_epoch = 0
        self.num_iter = 0
        self.log = []
        self.lr = tf.constant(self.learning_rate)

        # Compute the objective function.
        self.vae_loss, self.dis_loss = self.build_vae()

        vars = tf.trainable_variables()
        dis_vars = [v for v in vars if v.name.startswith('D/')]
        vae_vars = [v for v in vars if v.name.startswith('G/')]



        # Build a graph that trains the model with one batch of examples
        # and updates the model parameters.
        self.vae_opt = tf.train.AdamOptimizer(self.lr).minimize(self.vae_loss, var_list=vae_vars)
        self.dis_opt = tf.train.AdamOptimizer(self.lr).minimize(self.dis_loss, var_list=dis_vars)

        # Build an initialization operation to run.
        init = tf.initialize_all_variables()
        self.sess.run(init)
        if self.load:
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, self.load_file)

    def train(self):
        """ Train VAE for a number of steps."""
        start_time = time.time()


        num_batches = int(self.n_samples / self.batch_size)

        for epoch in xrange(self.num_epochs):
            avg_loss_value = 0.

            epoch_start = time.time()

            for b in xrange(num_batches):
                # Get images from the mnist dataset.
                batch_images, batch_info = self.input()

                # Sample a batch of eps from standard normal distribution.
                batch_eps = np.random.randn(self.batch_size,self.z_dim)
                batch_eps2 = np.random.randn(self.batch_size,self.z_dim)

                # Run a step of adam optimization and loss computation.

                _, vae_loss, x_mean, x_fresh = self.sess.run(
                    [self.vae_opt, self.vae_loss, self.x_mean, self.x_fresh],
                                            feed_dict = {self.x: batch_images,
                                                        self.y: batch_info,
                                                        self.eps: batch_eps,
                                                        self.z_fresh: batch_eps2,
                                                        self.lr: self.learning_rate})
                _, dis_loss = self.sess.run([self.dis_opt, self.dis_loss],
                                            feed_dict = {self.x: batch_images,
                                                        #self.eps: batch_eps,
                                                        self.x_fresh: x_fresh,
                                                        self.x_mean: x_mean,
                                                        self.lr: self.learning_rate})
                loss_value = vae_loss + dis_loss

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
        #self.save(epoch)

    def save(self,epoch):
        self.saver = tf.train.Saver()
        self.saver.save(self.sess, os.path.join(self.checkpoint_dir,
            self.checkpoint_name), global_step = epoch)

    def input(self):
        """
        This function reads in one batch of data.
        """
        if self.dataset == 'mnist':
            # Extract images and labels (currently useless) from the next batch.
            batch_images, batch_info = self.mnist.train.next_batch(self.batch_size)

            return batch_images, batch_info

        batch_images, batch_info = self.dataset.next_batch(self.batch_size)
        return batch_images, batch_info

    def build_vae(self):
        """
        This function builds up VAE from encoder and decoder function
        in the global environment.
        """
        # Add a placeholder for one batch of images
        with tf.variable_scope('G'):
            self.x = tf.placeholder(tf.float32,[self.batch_size,self.x_dim],
                                        name = 'x')

            self.y = tf.placeholder(tf.float32,[self.batch_size, self.cond_info_dim], name='y')

            # Create a placeholder for eps.
            self.eps = tf.placeholder(tf.float32,[self.batch_size, self.z_dim], name = 'eps')

            # Construct the mean and the variance of q(z|x).
            # use this line for fc
            self.z_mean, self.z_log_sigma2 = self.build_encoder(tf.concat(1, (self.x, self.y)), self.z_dim)
            # use this line for conv
            #self.z_mean, self.z_log_sigma2 = self.build_encoder(self.x, self.y, self.z_dim)


            # Compute z from eps and z_mean, z_sigma2.
            self.z = tf.add(self.z_mean, tf.mul(tf.sqrt(tf.exp(self.z_log_sigma2)), self.eps))
            # Construct the mean of the Bernoulli distribution p(x|z).
            self.x_mean = self.build_decoder(tf.concat(1, (self.z, self.y)), self.x_dim)
            # Compute the loss from decoder (empirically).
            self.decoder_loss = -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_mean) \
                               + (1 - self.x) * tf.log(1e-10 + 1 - self.x_mean),
                               1)
            # Compute the loss from encoder (analytically).
            self.encoder_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma2
                                               - tf.square(self.z_mean)
                                               - tf.exp(self.z_log_sigma2), 1)

            self.z_fresh = tf.placeholder(tf.float32,[self.batch_size, self.z_dim], name = 'z_fresh')
            self.x_fresh = self.build_decoder(tf.concat(1, (self.z_fresh, self.y)), self.x_dim, reuse=True)

        with tf.variable_scope('D'):
            # true image
            self.dis_real = self.build_discriminator(self.x)
            # reconstruction
            self.dis_fake_recon = self.build_discriminator(self.x_mean, reuse=True)
            # fresh generation
            self.dis_fake_fresh = self.build_discriminator(self.x_fresh, reuse=True)

        dis_loss = tf.reduce_mean(-tf.log(self.dis_real+1e-10) - tf.log(1.-self.dis_fake_fresh+1e-10) - tf.log(1.-self.dis_fake_recon+1e-10))


        # Add up to the cost.
        vae_loss = tf.reduce_mean(self.encoder_loss + self.decoder_loss - tf.log(self.dis_fake_fresh+1e-10) - self.gamma * tf.log(self.dis_fake_recon+1e-10))

        return vae_loss, dis_loss

    def generate(self,num = 10, info = None, load = False):
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

        return self.sess.run(self.x_mean,\
                      feed_dict = {self.z:sampled_z, self.y:info})

    def get_code(self, img):
        return self.sess.run(self.z_mean, feed_dict={self.x:img})

