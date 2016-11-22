import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import os
from gumbel_ops import *
slim=tf.contrib.slim



class SSL_Gumbel():
    def __init__(self, sess, build_encoder1, build_encoder2, build_decoder, dataset_l, dataset_u,
        checkpoint_name = 'SSL_cat_checkpoint',
        batch_size = 100, z_dim = 64, x_dim = 784, y_dim = 10, alpha = 5500.,
        learning_rate = 0.001, num_epochs = 5,load = False,load_file = None,
        checkpoint_dir = '../notebook/checkpoints/', summaries_dir = 'gumbel_log/',
        tau0=1.0, anneal_rate=0.00003, min_temp=0.5, lr_decay=0.9):
        """
        Inputs:
        sess: TensorFlow session.
        build_encoder1, build_encoder2: A function that lays down the computational
        graph for the encoder. build_encoder1 only takes x,
        while build_encoder2 will take y and the output of build_encoder1
        build_decoder: A function that lays down the computational
        graph for the decoder.
        dataset_l: labeled dataset
        dataset_u: unlabeled dataset
        checkpoint_name: The name of the checkpoint file to be saved.
        batch_size: The number of samples in each batch.
        z_dim: the dimension of z.
        x_dim: the dimension of an image.
        y_dim: number of labels
        alpha: tuning parameter (see paper)
        (Currently, we only consider 28*28 = 784.)
        dataset: The filename of the dataset.
        (Currently we only consider mnist.)
        learning_rate: The learning rate of the Adam optimizer.
        num_epochs: The number of epochs.

        """
        self.sess = sess
        self.build_encoder1 = build_encoder1
        self.build_encoder2 = build_encoder2
        self.build_decoder = build_decoder
        self.logaset_l = dataset_l
        self.logaset_u = dataset_u
        self.checkpoint_name = checkpoint_name
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.load = load
        self.load_file = load_file
        self.checkpoint_dir = checkpoint_dir
        self.alpha = alpha
        self.summaries_dir = summaries_dir
        self.tau0=tau0
        self.anneal_rate=anneal_rate
        self.min_temp=min_temp
        self.lr_decay=lr_decay
        # if dataset == 'mnist':
        #     # Load MNIST data in a format suited for tensorflow.
        #     self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        #     self.n_samples = self.mnist.train.num_examples
        #if dataset == 'mnist' and load == False:
            # Load MNIST data in a format suited for tensorflow.
            #self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
            #self.n_samples = self.mnist.train.num_examples



        self.n_samples_l = dataset_l.num_examples
        self.n_samples_u = dataset_u.num_examples

        global_step = tf.Variable(0, trainable = False)

        # Compute the objective function.
        self.loss_l = self.build_vae_l()
        self.loss_u = self.build_vae_u()
        #self.summary_l = tf.scalar_summary('loss for labeled data', self.loss_l)
        #self.summary_u = tf.scalar_summary('loss for unlabeled data', self.loss_u)

        #merged = tf.merge_all_summaries()
        #self.summary_writer = tf.train.SummaryWriter(self.summaries_dir,self.sess.graph)

        self.num_iter_l = 0
        self.num_iter_u = 0
        self.num_iter = 0
        self.lr = tf.constant(self.learning_rate)
        self.np_temp = self.tau0


        # Get optimizers
        self.optimum_l = tf.train.AdamOptimizer(self.lr).minimize(self.loss_l, global_step=global_step, var_list=slim.get_model_variables())
        self.optimum_u = tf.train.AdamOptimizer(self.lr).minimize(self.loss_u, global_step=global_step, var_list=slim.get_model_variables())


        # Lay down the graph for computing accuracy.

        #self.y_ = tf.placeholder(tf.float32,[self.batch_size, self.y_dim], name = 'y_')

        #with tf.name_scope('accuracy'):
            #with tf.name_scope('correct_prediction'):
                #correct_prediction = tf.equal(tf.argmax(self.encoder_y_logit_l, 1), tf.argmax(self.y_, 1))
            #with tf.name_scope('accuracy'):
                #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            #self.summary_acc = tf.scalar_summary('training accuracy', accuracy)

        # Initialize
        init = tf.initialize_all_variables()
        self.sess.run(init)

        self.log=[]


        if self.load:
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, self.load_file)


    def train(self):
        """ Train VAE for a number of steps."""

        num_batches_l = int(self.n_samples_l / self.batch_size)
        num_batches_u = int(self.n_samples_u / self.batch_size)

        for epoch in xrange(self.num_epochs):
            avg_loss_value = 0.

            # Process labeled batch
            for b in xrange(num_batches_l):
                # Get images and labels
                batch_x_l, batch_y_l = self.input_l()

                # Sample a batch of eps from standard normal distribution.
                batch_eps_l = np.random.randn(self.batch_size, self.z_dim)

                # Run a step of adam optimization and loss computation.
                start_time = time.time()
                _, loss_value = self.sess.run([self.optimum_l, self.loss_l],
                                            feed_dict = {self.x_l: batch_x_l,
                                                        self.y_l: batch_y_l,
                                                        self.batch_eps_l: batch_eps_l#,
                                                        #self.train_phase_l:True,
                                                        #self.train_phase_u:True
                                                        })
                duration = time.time() - start_time



                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                avg_loss_value += loss_value / self.n_samples_l * self.batch_size
                self.num_iter_l += 1
                self.num_iter += 1

                if self.num_iter % 100 == 1:
                    self.log.append([self.num_iter, self.np_temp, loss_value / self.n_samples_l * self.batch_size])

                if self.num_iter % 1000 == 1:
                    self.learning_rate *= self.lr_decay

                if b % 10 == 0:
                    pass
                    # Add training loss_l to self.summary_writer.
                    #summary = self.sess.run(self.summary_l, feed_dict = {self.x_l: batch_x_l,
                                                            #self.y_l: batch_y_l,
                                                            #self.batch_eps_l: batch_eps_l,
                                                            #self.train_phase_l:True,
                                                            #self.train_phase_u:True})

                    #self.summary_writer.add_summary(summary, b + epoch * num_batches_l)

                # Add training accuracy to self.summary_writer.
                if b % 10 == 0:
                    pass
                    #summary = self.sess.run(self.summary_acc,feed_dict = {self.x_l: batch_x_l,
                            #self.y_: batch_y_l, self.train_phase_l:False, self.train_phase_u:False})

                    #self.summary_writer.add_summary(summary, b + epoch * num_batches_l)


            for b in xrange(num_batches_u):
                # Get unlabeled images
                batch_x_u = self.input_u()

                # Sample a batch of eps from standard normal distribution.
                batch_eps_u = np.random.randn(self.batch_size, self.z_dim)

                # Run a step of adam optimization and loss computation.
                start_time = time.time()
                # sample label y from the distribution q(y|x)
                # y_u = self.sess.run(self.encoder_y_logit_u, feed_dict = {self.x_u: batch_x_u, self.train_phase_l:True, self.train_phase_u:True})
                # then input image and sampled label
                _, loss_value = self.sess.run([self.optimum_u, self.loss_u],
                                            feed_dict = {self.x_u: batch_x_u,
                                                        # self.y_u: y_u,
                                                        self.batch_eps_u: batch_eps_u,
                                                        #self.train_phase_l:True,
                                                        #self.train_phase_u:True,
                                                        self.tau: self.np_temp})
                duration = time.time() - start_time


                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                avg_loss_value += loss_value / self.n_samples_u * self.batch_size
                self.num_iter_u += 1
                self.num_iter += 1

                if self.num_iter % 100 == 1:
                    self.log.append([self.num_iter, self.np_temp, loss_value / self.n_samples_l * self.batch_size])
                if self.num_iter_u % 1000 == 1:
                    self.np_temp = np.maximum(self.tau0*np.exp(-self.anneal_rate*self.num_iter_u), self.min_temp)
                if self.num_iter % 1000 == 1:
                    self.learning_rate *= self.lr_decay

                if b % 10 == 0:
                    pass
                    # Add training loss_l to self.summary_writer.
                    #summary = self.sess.run(self.summary_u, feed_dict = {self.x_u: batch_x_u,
                                                        #self.y_u: y_u,
                                                        #self.batch_eps_u: batch_eps_u,
                                                        #self.train_phase_l:False,
                                                        #self.train_phase_u:False})

                    #self.summary_writer.add_summary(summary, b + epoch * num_batches_u)


            print 'Epoch {} loss: {}'.format(epoch + 1, avg_loss_value)

        #self.summary_writer.close()
        self.save(epoch)

    def save(self,epoch):
        self.saver = tf.train.Saver()
        #self.saver.save(self.sess, os.path.join(self.checkpoint_dir, self.checkpoint_name), global_step = epoch)
        self.saver.save(self.sess, os.path.join(self.checkpoint_dir, self.checkpoint_name))


    def input_l(self):
        """
        This function reads in one batch of labeled data
        """
        x_l, y_l = self.logaset_l.next_batch(self.batch_size)
        return x_l, y_l
        #if self.logaset == 'mnist':
            # Extract images and labels (currently useless) from the next batch.
            #batch_images, batch_labels = self.mnist.train.next_batch(self.batch_size)

            #return batch_images, batch_labels
    def input_u(self):
        """
        This function reads in one batch of unlabeled data.
        """

        x_u, _ = self.logaset_u.next_batch(self.batch_size)
        return x_u
        #if self.logaset == 'mnist':
            # Extract images and labels (currently useless) from the next batch.
            #batch_images, batch_labels = self.mnist.train.next_batch(self.batch_size)

            #return batch_images, batch_labels

    def build_vae_l(self):
        #print "building vae l"
        """
        This function builds up the VAE for labeled data from encoder and decoder function
        in the global environment.
        """
        # Add a placeholder for one batch of images and labels
        self.x_l = tf.placeholder(tf.float32,[self.batch_size, self.x_dim], name = 'x_l')
        self.y_l = tf.placeholder(tf.float32,[self.batch_size, self.y_dim], name = 'y_l')

        # Create a placeholder for eps.
        self.batch_eps_l = tf.placeholder(tf.float32,[self.batch_size, self.z_dim], name = 'eps_l')

        # boolean for training
        #self.train_phase_l = tf.placeholder(tf.bool, name='train_phase_l')


        # Construct the mean and the variance of q(z|x).
        self.y_logit_l = self.build_encoder1(self.x_l, self.y_dim, batch_size=self.batch_size)
        self.y_prob_l = tf.nn.softmax(self.y_logit_l)


        # self.encoder_log_sigma_sq_l, self.encoder_y_logit_l, h1_l = self.build_encoder1(self.x_l, self.z_dim, self.y_dim, train_phase=self.train_phase_l)
        #self.encoder_mu_l = self.build_encoder2(h1_l, self.y_l, self.z_dim, train_phase=self.train_phase_l)
        self.z_mu_l, self.z_log_sigma2_l = self.build_encoder2(self.x_l, self.y_l, self.z_dim, batch_size=self.batch_size)

        # self.encoder_y_prob_l = tf.nn.softmax(self.encoder_y_logit_l)

        # Compute z from eps and z_mean, z_sigma2.
        self.z_l = tf.add(self.z_mu_l, tf.mul(tf.sqrt(tf.exp(self.z_log_sigma2_l)), self.batch_eps_l))

        # Construct the mean of the Bernoulli distribution p(x|z).
        self.x_logit_l = self.build_decoder(self.z_l, self.y_l, self.x_dim, batch_size=self.batch_size)
        #self.decoder_mean_l = self.build_decoder(self.z_l, self.y_l, self.x_dim, train_phase=self.train_phase_l)
        self.x_prob_l = 1./(1. + tf.exp(-self.x_logit_l))

        # Compute the loss from decoder (empirically).
        self.decoder_loss_l = -tf.reduce_sum(self.x_l * tf.log(1e-10 + self.x_prob_l) \
                           + (1 - self.x_l) * tf.log(1e-10 + 1 - self.x_prob_l),
                           1)
        # Compute the loss from encoder (analytically).
        self.encoder_loss_l = -0.5 * tf.reduce_sum(1 + self.z_log_sigma2_l
                                           - tf.square(self.z_mu_l)
                                           - tf.exp(self.z_log_sigma2_l), 1)

        # - log p(y)
        self.label_loss_l = -self.batch_size * tf.log(1e-10 + 1./self.y_dim)

        # classification loss, weighted by alpha
        self.classification_loss_l = - self.alpha * tf.log(1e-10 + tf.reduce_sum(self.y_prob_l * self.y_l, 1))

        # Add up to the cost.
        self.cost_l = tf.reduce_mean(self.encoder_loss_l + self.decoder_loss_l + self.label_loss_l + self.classification_loss_l)

        return self.cost_l

    def build_vae_u(self):
        #print "building vae_u"
        """
        This function builds up VAE for unlabeled data from encoder and decoder function
        in the global environment.
        """
        # Add a placeholder for one batch of images
        self.x_u = tf.placeholder(tf.float32,[self.batch_size, self.x_dim], name = 'x_u')
        # self.y_u = tf.placeholder(tf.float32,[self.batch_size, self.y_dim], name = 'y_u')

        # Create a placeholder for eps.
        self.batch_eps_u = tf.placeholder(tf.float32,[self.batch_size, self.z_dim], name = 'eps_u')

        self.tau = tf.Variable(5.0,name="temperature")

        # boolean for training flag in batch normalization
        # self.train_phase_u = tf.placeholder(tf.bool, name='train_phase_u')

        # Construct the mean and the variance of q(z|x).
        self.y_logit_u = self.build_encoder1(self.x_u, self.y_dim, reuse=True, batch_size=self.batch_size)
        self.y_prob_u = tf.nn.softmax(self.y_logit_u)
        #self.encoder_log_sigma_sq_u, self.encoder_y_logit_u, h1_u = self.build_encoder1(self.x_u, self.z_dim, self.y_dim, reuse=True, train_phase=self.train_phase_u)

        #self.encoder_y_prob_u = tf.nn.softmax(self.encoder_y_logit_u)

        self.y_u = gumbel_softmax(self.y_logit_u+1e-10, self.tau, hard=False)

        self.z_mu_u, self.z_log_sigma2_u = self.build_encoder2(self.x_u, self.y_u, self.z_dim, reuse=True, batch_size=self.batch_size)

        # self.encoder_mu_u = self.build_encoder2(h1_u, self.y_u, self.z_dim, reuse=True, train_phase=self.train_phase_u)

        # Compute z from eps and z_mean, z_sigma2.
        self.z_u = tf.add(self.z_mu_u, tf.mul(tf.sqrt(tf.exp(self.z_log_sigma2_u)), self.batch_eps_u))

        # Construct the mean of the Bernoulli distribution p(x|z).
        self.x_logit_u = self.build_decoder(self.z_u, self.y_u, self.x_dim, reuse=True, batch_size=self.batch_size)
        #self.decoder_mean_u = self.build_decoder(self.z_u, self.y_u, self.x_dim, reuse=True, train_phase=self.train_phase_u)
        self.x_prob_u = 1./(1. + tf.exp(-self.x_logit_u))

        # Compute the loss from decoder (empirically).

        self.decoder_loss_u = -tf.reduce_sum(self.x_u * tf.log(1e-10 + self.x_prob_u) \
                           + (1 - self.x_u) * tf.log(1e-10 + 1 - self.x_prob_u),
                           1)
        # Compute the loss from encoder (analytically).
        self.encoder_loss_u = -0.5 * tf.reduce_sum(1 + self.z_log_sigma2_u
                                           - tf.square(self.z_mu_u)
                                           - tf.exp(self.z_log_sigma2_u), 1)

        # - log p(y)
        self.label_loss_u = -self.batch_size * tf.log(1e-10 + 1./self.y_dim)

        # extra entropy term H(q(y|x)) in loss for unlabeled data
        self.classification_loss_u = - tf.log(1e-10 + tf.reduce_sum(self.y_prob_u * self.y_u, 1))

        # Add up to the cost.
        self.cost_u = tf.reduce_mean(self.encoder_loss_u + self.decoder_loss_u + self.label_loss_u + self.classification_loss_u)

        return self.cost_u

    def generate(self,num = 10, labels = None):
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

        if labels is None:
            _, labels = self.input_l()

        return self.sess.run(self.decoder_mean_l, feed_dict = {self.z_l: sampled_z, self.y_l: labels, self.train_phase_l:False, self.train_phase_u:False})

    def classify(self, x):
        """
        Classify input x
        """
        n = x.shape[0]
        assert n <= self.batch_size, "Cannot classify more than batch size at one time."

        return np.argmax(self.sess.run(self.y_logit_l, feed_dict = {self.x_l: x,
            #self.train_phase_l:False, self.train_phase_u:False
            }), axis=1)
