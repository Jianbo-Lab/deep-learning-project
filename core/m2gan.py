import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import os

"""
NOTE: the call to build_encoder in build_vae_l and build_vae_u needs to modified
depending on whether the encoder/decoder is fully connected or convolutional
ctrl+F "TOGGLE" to find the line in this file
"""


class SSL_M2():
    def __init__(self, sess, build_encoder1, build_encoder2, build_decoder, build_discriminator, dataset_l, dataset_u,
        checkpoint_name = 'SSL_M2_checkpoint',
        batch_size = 100, z_dim = 20, x_dim = 784, y_dim = 10,  
        learning_rate = 0.001, lr_decay=0.95, lr_decay_freq=1000, num_epochs = 5,load = False,load_file = None,
        checkpoint_dir = '../notebook/checkpoints/', summaries_dir = 'm2_logs/', x_width=28,
        beta = 1,gamma = 1,theta = 5500,alpha = 1):
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
        self.dataset_l = dataset_l
        self.dataset_u = dataset_u
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
        self.lr_decay=lr_decay
        self.lr_decay_freq=lr_decay_freq
        self.x_width=x_width
        self.beta = beta
        self.theta = theta
        self.gamma = gamma
        self.build_discriminator = build_discriminator
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
        self.vae_loss_l,self.dis_loss_l = self.build_vae_l()
        self.vae_loss_u,self.dis_loss_u = self.build_vae_u()  

        """
        self.summary_l = tf.scalar_summary('loss for labeled data', self.loss_l)
        self.summary_u = tf.scalar_summary('loss for unlabeled data', self.loss_u)

        merged = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter(self.summaries_dir,self.sess.graph)
        """

        self.num_epoch = 0
        self.num_iter = 0
        self.log = []
        self.lr = tf.constant(self.learning_rate)


        vars = tf.trainable_variables()
        dis_vars = [v for v in vars if v.name.startswith('D/')]
        vae_vars = [v for v in vars if v.name.startswith('G/')]

        # Get optimizers
        self.vae_opt_l = tf.train.AdamOptimizer(self.lr).minimize(self.vae_loss_l, global_step=global_step, var_list=vae_vars)
        self.vae_opt_u = tf.train.AdamOptimizer(self.lr).minimize(self.vae_loss_u, global_step=global_step, var_list=vae_vars)
        self.dis_opt_l = tf.train.AdamOptimizer(self.lr).minimize(self.dis_loss_l, global_step=global_step, var_list=dis_vars)
        self.dis_opt_u = tf.train.AdamOptimizer(self.lr).minimize(self.dis_loss_u, global_step=global_step, var_list=dis_vars)

        # Lay down the graph for computing accuracy.

        """
        self.y_ = tf.placeholder(tf.float32,[self.batch_size, self.y_dim], name = 'y_')

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(self.encoder_y_prob_l, 1), tf.argmax(self.y_, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.summary_acc = tf.scalar_summary('training accuracy', accuracy)
        """

        # Initialize
        init = tf.initialize_all_variables()
        self.sess.run(init)


        if self.load:
            #{v.op.name: v for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope='G')}
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, self.load_file)

    def train(self):
        """ Train VAE for a number of steps."""

        start_time = time.time()

        num_batches_l = int(self.n_samples_l / self.batch_size)
        num_batches_u = int(self.n_samples_u / self.batch_size)

        for epoch in xrange(self.num_epochs):
            avg_loss_value = 0.

            epoch_start = time.time()

            # Process labeled batch
            for b in xrange(num_batches_l):
                # Get images and labels
                batch_x_l, batch_y_l = self.input_l()

                # Sample a batch of eps from standard normal distribution.
                batch_eps_l = np.random.randn(self.batch_size, self.z_dim)
                batch_eps2 = np.random.randn(self.batch_size,self.z_dim)
                label_extra_real = np.hstack((batch_y_l,np.tile(0,(self.batch_size,1)))).astype(np.int32)
                label_extra_fresh = np.hstack((np.tile(0,(self.batch_size,10)),np.tile(1,(self.batch_size,1)))).astype(np.int32)
                
                # Run a step of adam optimization and loss computation.
                start_time = time.time()
                _, vae_loss_l,x_fresh_l = self.sess.run([self.vae_opt_l, self.vae_loss_l,self.x_fresh_l],
                                            feed_dict = {self.x_l: batch_x_l,
                                                        self.y_l: batch_y_l,
                                                        self.batch_eps_l: batch_eps_l,
                                                        self.z_fresh_l: batch_eps2,
                                                        self.train_phase_l:True, 
                                                        self.label_extra_real_l: label_extra_real, 
                                                        self.lr: self.learning_rate})
                _, dis_loss_l = self.sess.run([self.dis_opt_l, self.dis_loss_l],
                                            feed_dict = {self.x_l: batch_x_l, 
                                                        self.x_fresh_l: x_fresh_l, 
                                                        self.lr: self.learning_rate,
                                                        self.train_phase_l: True, 
                                                        self.label_extra_real_l: label_extra_real,
                                                        self.label_extra_fresh_l: label_extra_fresh})
                duration = time.time() - start_time


 
 

                self.num_iter += 1

                print 'Epoch {} Labeled batch: {} Generator loss: {} Discriminator loss: {}'.format(self.num_epoch, b, vae_loss_l, dis_loss_l)
                #print 'Epoch {} Encoder loss: {} Decoder loss: {} Fake loss: {}'.format(self.num_epoch, encoder_loss, decoder_loss, fake_loss)
                
                if self.num_iter % self.lr_decay_freq == 1:
                    self.learning_rate *= self.lr_decay


                """
                if b % 10 == 0:
                    # Add training loss_l to self.summary_writer.
                    summary = self.sess.run(self.summary_l, feed_dict = {self.x_l: batch_x_l,
                                                            self.y_l: batch_y_l,
                                                            self.batch_eps_l: batch_eps_l,
                                                            self.train_phase_l:True,
                                                            self.train_phase_u:True})

                    self.summary_writer.add_summary(summary, b + epoch * num_batches_l)

                # Add training accuracy to self.summary_writer.
                if b % 10 == 0:
                    summary = self.sess.run(self.summary_acc,feed_dict = {self.x_l: batch_x_l,
                            self.y_: batch_y_l, self.train_phase_l:False, self.train_phase_u:False})

                    self.summary_writer.add_summary(summary, b + epoch * num_batches_l)
                """


            for b in xrange(num_batches_u):
                # Get unlabeled images
                batch_x_u = self.input_u()

                # Sample a batch of eps from standard normal distribution.
                batch_eps_u = np.random.randn(self.batch_size, self.z_dim)
                batch_eps2 = np.random.randn(self.batch_size,self.z_dim)

                # Run a step of adam optimization and loss computation.
                start_time = time.time()
                # sample label y from the distribution q(y|x)
                y_u = self.sess.run(self.encoder_y_prob_u, feed_dict = {self.x_u: batch_x_u, self.train_phase_l:True, self.train_phase_u:True})
                label_extra_real = np.hstack((y_u,np.tile(0,(self.batch_size,1)))).astype(np.int32)
                label_extra_fresh = np.hstack((np.tile(0,(self.batch_size,10)),np.tile(1,(self.batch_size,1)))).astype(np.int32)
                
                # then input image and sampled label
                _, vae_loss_u,x_fresh_u = self.sess.run([self.vae_opt_u, self.vae_loss_u,self.x_fresh_u],
                                            feed_dict = {self.x_u: batch_x_u,
                                                        self.y_u: y_u,
                                                        self.batch_eps_u: batch_eps_u, 
                                                        self.z_fresh_u: batch_eps2,
                                                        self.train_phase_u:True,
                                                        self.label_extra_real_u: label_extra_real,
                                                        self.lr: self.learning_rate})
                _, dis_loss_u = self.sess.run([self.dis_opt_u, self.dis_loss_u],
                                            feed_dict = {self.x_u: batch_x_u,
                                                        #self.eps: batch_eps,
                                                        self.x_fresh_u: x_fresh_u, 
                                                        self.lr: self.learning_rate,
                                                        self.train_phase_u: True,
                                                        self.label_extra_real_u: label_extra_real,
                                                        self.label_extra_fresh_u: label_extra_fresh})
                duration = time.time() - start_time
#
                print 'Epoch {} Unlabeled batch {} Generator loss: {} Discriminator loss: {}'.format(self.num_epoch, b, vae_loss_u, dis_loss_u)
                # print 'Epoch {} Encoder loss: {} Decoder loss: {} Fake loss: {}'.format(self.num_epoch, encoder_loss, decoder_loss, fake_loss)
                
 

                self.num_iter += 1
 
                if self.num_iter % self.lr_decay_freq == 1:
                    self.learning_rate *= self.lr_decay

                """
                if b % 10 == 0:
                    # Add training loss_l to self.summary_writer.
                    summary = self.sess.run(self.summary_u, feed_dict = {self.x_u: batch_x_u,
                                                        self.y_u: y_u,
                                                        self.batch_eps_u: batch_eps_u,
                                                        self.train_phase_l:False,
                                                        self.train_phase_u:False})

                    self.summary_writer.add_summary(summary, b + epoch * num_batches_u)
                """

            self.num_epoch += 1
            epoch_end = time.time() 
            #self.save(self.num_epoch)
        end_time = time.time()
        print '{} min'.format((end_time-start_time)/60.)
        #self.summary_writer.close()
        self.save(epoch)

    def save(self,epoch):
        self.saver = tf.train.Saver()
        #self.saver.save(self.sess, os.path.join(self.checkpoint_dir, self.checkpoint_name), global_step = epoch)
        self.saver.save(self.sess, os.path.join(self.checkpoint_dir, self.checkpoint_name), global_step=epoch)


    def input_l(self):
        """
        This function reads in one batch of labeled data
        """
        x_l, y_l = self.dataset_l.next_batch(self.batch_size)
        return x_l, y_l
        #if self.dataset == 'mnist':
            # Extract images and labels (currently useless) from the next batch.
            #batch_images, batch_labels = self.mnist.train.next_batch(self.batch_size)

            #return batch_images, batch_labels
    def input_u(self):
        """
        This function reads in one batch of unlabeled data.
        """

        x_u, _ = self.dataset_u.next_batch(self.batch_size)
        return x_u
        #if self.dataset == 'mnist':
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
        with tf.variable_scope('G'):
            self.x_l = tf.placeholder(tf.float32,[self.batch_size, self.x_dim], name = 'x_l')
            self.y_l = tf.placeholder(tf.float32,[self.batch_size, self.y_dim], name = 'y_l')

            # Create a placeholder for eps.
            self.batch_eps_l = tf.placeholder(tf.float32,[self.batch_size, self.z_dim], name = 'eps_l')

            # boolean for training
            self.train_phase_l = tf.placeholder(tf.bool, name='train_phase_l')


            # Construct the mean and the variance of q(z|x).
            """
            === TOGGLE ===
            NOTE: the call to build_encoder in build_vae_l needs to modified
            depending on whether the encoder/decoder is fully connected or convolutional
            """
            # fc
            #self.encoder_log_sigma_sq_l, self.encoder_y_prob_l, h1_l = self.build_encoder1(self.x_l, self.z_dim, self.y_dim, train_phase=self.train_phase_l)
            # conv
            self.encoder_log_sigma_sq_l, self.encoder_y_prob_l, h1_l = self.build_encoder1(self.x_l, self.z_dim, self.y_dim, train_phase=self.train_phase_l, x_width=self.x_width)

            self.encoder_mu_l = self.build_encoder2(h1_l, self.y_l, self.z_dim, train_phase=self.train_phase_l)

            # Compute z from eps and z_mean, z_sigma2.
            self.batch_z_l = tf.add(self.encoder_mu_l, tf.mul(tf.sqrt(tf.exp(self.encoder_log_sigma_sq_l)), self.batch_eps_l))

            # Construct the mean of the Bernoulli distribution p(x|z).
            self.decoder_mean_l = self.build_decoder(tf.concat(1, (self.batch_z_l, self.y_l)), self.x_dim, train_phase=self.train_phase_l)

            self.z_fresh_l = tf.placeholder(tf.float32,[self.batch_size, self.z_dim], name = 'z_fresh_l')
            self.x_fresh_l = self.build_decoder(tf.concat(1, (self.z_fresh_l, self.y_l)), self.x_dim, reuse=True)






           
        with tf.variable_scope('D'):

            # true image
            self.dis_real_l,self.lth_layer_real_l = self.build_discriminator(self.x_l, train_phase = True)

            # # reconstruction
            self.dis_fake_recon_l,self.lth_layer_recon_l = self.build_discriminator(self.decoder_mean_l, reuse=True, train_phase = True)

            # fresh generation
            self.dis_fake_fresh_l, self.lth_layer_fresh_l = self.build_discriminator(self.x_fresh_l, reuse=True, train_phase = True) 

 
        # Compute the loss from encoder (analytically).
        self.encoder_loss_l = -0.5 * tf.reduce_sum(1 + self.encoder_log_sigma_sq_l
                                           - tf.square(self.encoder_mu_l)
                                           - tf.exp(self.encoder_log_sigma_sq_l), 1)
        self.enc_loss_l = tf.reduce_mean(self.encoder_loss_l)
        self.decoder_loss_l = tf.reduce_sum(tf.square(self.lth_layer_real_l - self.lth_layer_recon_l)) 

        self.label_extra_real_l = tf.placeholder(tf.int32,[self.batch_size, 11], name='label_extra_real_l')
        self.label_extra_fresh_l = tf.placeholder(tf.int32,[self.batch_size, 11], name='label_extra_fresh_l')

        self.cross_entropy_real_l = tf.nn.softmax_cross_entropy_with_logits(self.dis_real_l, 
            self.label_extra_real_l, name='xentropy_real_l')

        self.cross_entropy_fresh_l = tf.nn.softmax_cross_entropy_with_logits(self.dis_fake_fresh_l, 
            self.label_extra_fresh_l, name='xentropy_fresh_l')


        dis_loss_l = tf.reduce_mean(self.cross_entropy_real_l + self.cross_entropy_fresh_l, name='xentropy_mean_l') 
    
        self.cross_entropy_fresh_fake_l = tf.nn.softmax_cross_entropy_with_logits(self.dis_fake_fresh_l, self.label_extra_real_l, name='xentropy_fresh_fake_l')
        self.g_loss_l = tf.reduce_mean(self.cross_entropy_fresh_fake_l) 
 

        # classification loss, weighted by alpha
        self.classification_loss_l = - tf.log(1e-10 + tf.reduce_sum(self.encoder_y_prob_l * self.y_l, 1))

        # Add up to the cost. 
        vae_loss_l = self.alpha * self.enc_loss_l + self.beta * self.decoder_loss_l + self.gamma * self.g_loss_l + self.theta * tf.reduce_mean(self.classification_loss_l) 
        
        return vae_loss_l,dis_loss_l

    def build_vae_u(self):
        #print "building vae_u"
        """
        This function builds up VAE for unlabeled data from encoder and decoder function
        in the global environment.
        """
        # Add a placeholder for one batch of images
        with tf.variable_scope('G'):
            self.x_u = tf.placeholder(tf.float32,[self.batch_size, self.x_dim], name = 'x_u')
            self.y_u = tf.placeholder(tf.float32,[self.batch_size, self.y_dim], name = 'y_u')

            # Create a placeholder for eps.
            self.batch_eps_u = tf.placeholder(tf.float32,[self.batch_size, self.z_dim], name = 'eps_u')

            # boolean for training flag in batch normalization
            self.train_phase_u = tf.placeholder(tf.bool, name='train_phase_u')

            # Construct the mean and the variance of q(z|x).
            """
            === TOGGLE ===
            NOTE: the call to build_encoder in build_vae_u needs to modified
            depending on whether the encoder/decoder is fully connected or convolutional
            """
            # fc:
            #self.encoder_log_sigma_sq_u, self.encoder_y_prob_u, h1_u = self.build_encoder1(self.x_u, self.z_dim, self.y_dim, reuse=True, train_phase=self.train_phase_u)
            #conv:
            self.encoder_log_sigma_sq_u, self.encoder_y_prob_u, h1_u = self.build_encoder1(self.x_u, self.z_dim, self.y_dim, reuse=True, train_phase=self.train_phase_u, x_width=self.x_width)


            self.encoder_mu_u = self.build_encoder2(h1_u, self.y_u, self.z_dim, reuse=True, train_phase=self.train_phase_u)



            # Compute z from eps and z_mean, z_sigma2.
            self.batch_z_u = tf.add(self.encoder_mu_u, tf.mul(tf.sqrt(tf.exp(self.encoder_log_sigma_sq_u)), self.batch_eps_u))

            # Construct the mean of the Bernoulli distribution p(x|z).
            self.decoder_mean_u = self.build_decoder(tf.concat(1, (self.batch_z_u, self.y_u)), self.x_dim, reuse=True, train_phase=self.train_phase_u)

            self.z_fresh_u = tf.placeholder(tf.float32,[self.batch_size, self.z_dim], name = 'z_fresh_u')
            self.x_fresh_u = self.build_decoder(tf.concat(1, (self.z_fresh_u, self.y_u)), self.x_dim, reuse=True)





           
        with tf.variable_scope('D'):

            # true image
            self.dis_real_u,self.lth_layer_real_u = self.build_discriminator(self.x_u,reuse=True, train_phase = True)

            # # reconstruction
            self.dis_fake_recon_u,self.lth_layer_recon_u = self.build_discriminator(self.decoder_mean_u, reuse=True, train_phase = True)

            # fresh generation
            self.dis_fake_fresh_u, self.lth_layer_fresh_u = self.build_discriminator(self.x_fresh_u, reuse=True, train_phase = True) 


        self.encoder_loss_u = -0.5 * tf.reduce_sum(1 + self.encoder_log_sigma_sq_u
                                           - tf.square(self.encoder_mu_u)
                                           - tf.exp(self.encoder_log_sigma_sq_u), 1)
        self.enc_loss_u = tf.reduce_mean(self.encoder_loss_u)
        self.decoder_loss_u = tf.reduce_sum(tf.square(self.lth_layer_real_u - self.lth_layer_recon_u)) 

        self.label_extra_real_u = tf.placeholder(tf.int32,[self.batch_size, 11], name='label_extra_real_u')
        self.label_extra_fresh_u = tf.placeholder(tf.int32,[self.batch_size, 11], name='label_extra_fresh_u')

        self.cross_entropy_real_u = tf.nn.softmax_cross_entropy_with_logits(self.dis_real_u, 
            self.label_extra_real_u, name='xentropy_real_u')

        self.cross_entropy_fresh_u = tf.nn.softmax_cross_entropy_with_logits(self.dis_fake_fresh_u, 
            self.label_extra_fresh_u, name='xentropy_fresh_u')


        dis_loss_u = tf.reduce_mean(self.cross_entropy_real_u + self.cross_entropy_fresh_u, name='xentropy_mean_u') 
    
        self.cross_entropy_fresh_fake_u = tf.nn.softmax_cross_entropy_with_logits(self.dis_fake_fresh_u, self.label_extra_real_u, name='xentropy_fresh_fake_u')
        self.g_loss_u = tf.reduce_mean(self.cross_entropy_fresh_fake_u) 
 

        # classification loss, weighted by alpha
        self.classification_loss_u = - tf.log(1e-10 + tf.reduce_sum(self.encoder_y_prob_u * self.y_u, 1))

        # Add up to the cost. 
        vae_loss_u = self.alpha * self.enc_loss_u + self.beta * self.decoder_loss_u + self.gamma * self.g_loss_u + tf.reduce_mean(self.classification_loss_u) 
        
 

        return vae_loss_u,dis_loss_u

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

        return self.sess.run(self.decoder_mean_l, feed_dict = {self.batch_z_l: sampled_z, self.y_l: labels, self.train_phase_l:False, self.train_phase_u:False})

    def classify(self, x):
        """
        Classify input x
        """
        n = x.shape[0]
        assert n <= self.batch_size, "Cannot classify more than batch size at one time."

        return np.argmax(self.sess.run(self.encoder_y_prob_l, feed_dict = {self.x_l: x, self.train_phase_l:False, self.train_phase_u:False}), axis=1)

