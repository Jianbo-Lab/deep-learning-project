import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import os

"""
NOTE: the call to build_encoder and build_discriminator in build_vae needs to modified
depending on whether the encoder/decoder is fully connected or convolutional
ctrl+F "TOGGLE" to find the line in this file
"""

def sigmoid(x,shift,mult):
    """
    Using this sigmoid to discourage one network overpowering the other
    """
    return 1 / (1 + np.exp(-(x+shift)*mult))

class CVAEGAN():
    def __init__(self, sess, build_encoder, build_decoder, build_discriminator, checkpoint_name = 'vae_checkpoint',
        batch_size = 100, z_dim = 20, x_dim = 784, dataset = 'mnist', testset = None, cond_info_dim=10,
        learning_rate = 0.001, lr_decay=0.95, lr_decay_freq=1000, num_epochs = 5,load = False,load_file = None, alpha = 1., beta = 1.0, gamma=1., summaries_dir = './', checkpoint_dir = '../notebook/checkpoints/'):
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
        self.alpha = alpha # weight for the encoder loss.
        self.beta = beta # weight for the decoder loss. 
        self.summaries_dir = summaries_dir
        
        # if dataset == 'mnist':
        #     # Load MNIST data in a format suited for tensorflow.
        #     self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        #     self.n_samples = self.mnist.train.num_examples
        if dataset == 'mnist' and load == False:
            # Load MNIST data in a format suited for tensorflow.
            self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
            self.n_samples = self.mnist.train.num_examples
            self.test_size = 100 # self.mnist.test.num_examples
            self.xtest, self.ytest = self.mnist.test.images[:100], self.mnist.test.labels[:100]
        elif load == False:
            self.n_samples = dataset.num_examples
            self.dataset = dataset
            self.testset = testset
            self.test_size = 100 
            self.xtest, self.ytest = self.testset.images[:100], self.testset.labels[:100]

        global_step = tf.Variable(0, trainable = False)


        self.num_epoch = 0
        self.num_iter = 0
        self.log = []
        self.lr = tf.constant(self.learning_rate)

        self.build_vae()

        # Build an initialization operation to run.
        init = tf.initialize_all_variables()
        self.sess.run(init)
        if self.load:
            self.saver = tf.train.Saver({v.op.name: v for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope='G')})
            self.saver.restore(self.sess, self.load_file)

        # Add summary writer.
        self.summary_L_d = tf.scalar_summary('loss for discriminator', self.L_d)
        self.summary_L_g = tf.scalar_summary('loss for generator', self.L_g)
        self.summary_L_e = tf.scalar_summary('loss for encoder', self.L_e) 

        merged = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter(self.summaries_dir,self.sess.graph)


    def train(self):
        """ Train VAE for a number of steps."""
        start_time = time.time()
        d_real = 0
        d_fake = 0

        num_batches = int(self.n_samples / self.batch_size)

        for epoch in xrange(self.num_epochs):
            
            epoch_start = time.time()

            for b in xrange(num_batches):
                # Get images from the mnist dataset.
                batch_images, batch_info = self.input()
 

                #label_extra_real = tf.concat(1,[batch_info,tf.tile([[0]],[self.batch_size,1])])
                #label_extra_fresh = tf.concat(1,[tf.tile([[0]],[self.batch_size,self.cond_info_dim]),tf.tile([[1]],[self.batch_size,1])])
                
                # Run a step of adam optimization and loss computation.
                e_current_lr = self.learning_rate*sigmoid(np.mean(d_real),-.5,15)
                g_current_lr = self.learning_rate*sigmoid(np.mean(d_real),-.5,15)
                d_current_lr = self.learning_rate*sigmoid(np.mean(d_fake),-.5,15)
                _, _, _, dloss,gloss,eloss,d_fake,d_real = self.sess.run([
                        self.d_opt,self.g_opt,self.e_opt,  
                        self.L_d,self.L_g,self.L_e,
                        self.prob_fake,self.prob_real],
                       {self.lr_E: e_current_lr,
                        self.lr_G: g_current_lr,
                        self.lr_D: d_current_lr,
                        self.x: batch_images,
                        self.y: batch_info,
                        self.train_phase: True})

                loss_value = dloss+gloss+eloss

                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                # avg_gen_loss_value += vae_loss / self.n_samples * self.batch_size
                # avg_dis_loss_value += dis_loss / self.n_samples * self.batch_size 
                print 'Epoch {} num_batches {} Discriminator loss: {} Generator loss: {} Encoder loss: {}'.format(self.num_epoch,b,dloss,gloss,eloss)
                self.num_iter += 1

 
                if self.num_iter % self.lr_decay_freq == 1:
                    self.learning_rate *= self.lr_decay
                if b % 50 == 0:
 
                    # Add training loss_l to self.summary_writer.  
                    aa,bb,cc = self.sess.run([self.summary_L_d, self.summary_L_g,self.summary_L_e], feed_dict = {self.x: self.xtest, self.y: self.ytest, self.train_phase: True})  
                      

                    self.summary_writer.add_summary(aa, b + epoch * num_batches)
                    self.summary_writer.add_summary(bb, b + epoch * num_batches)
                    self.summary_writer.add_summary(cc, b + epoch * num_batches) 
            # Later we'll add summary and log files to record training procedures.
            # For now, we will satisfy with naive printing.
            self.num_epoch += 1
            epoch_end = time.time()
            # print 'Epoch {} Generator loss: {} Discriminator loss: {} (time: {} s)'.format(self.num_epoch, avg_gen_loss_value, avg_dis_loss_value, epoch_end-epoch_start)
        self.save(self.num_epoch)
        end_time = time.time()
        print '{} min'.format((end_time-start_time)/60)
        #self.save(epoch)

    def save(self,epoch):
        self.saver = tf.train.Saver({v.op.name: v for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope='G')})
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
    def inference(self, x,y):
        """
        Run the models. Called inference because it does the same thing as tensorflow's cifar tutorial
        """
        z_fresh =  tf.random_normal((self.batch_size, self.z_dim), 0, 1) # normal dist for GAN
        eps = tf.random_normal((self.batch_size, self.z_dim), 0, 1) # normal dist for VAE
 
            
        with tf.variable_scope('E'):
            z_mean, z_log_sigma2 = self.build_encoder(x, y, self.z_dim) 

        with tf.variable_scope('G'):
  
            # Compute z from eps and z_mean, z_sigma2.
            z = tf.add(z_mean, tf.mul(tf.sqrt(tf.exp(z_log_sigma2)), eps))

            # Construct the mean of the Bernoulli distribution p(x|z).
            x_mean = self.build_decoder(tf.concat(1, (z, y)), self.x_dim)
 
            x_fresh = self.build_decoder(tf.concat(1, (z_fresh, y)), self.x_dim, reuse=True,train_phase = self.train_phase)

        with tf.variable_scope('D'): 

            # true image
            dis_real,lth_layer_real = self.build_discriminator(x, train_phase = True)

            # # reconstruction
            dis_fake_recon,lth_layer_recon = self.build_discriminator(x_mean, reuse=True, train_phase = True)

            # fresh generation
            dis_fake_fresh, lth_layer_fresh = self.build_discriminator(x_fresh, reuse=True, train_phase = True) 
        return z_mean, z_log_sigma2, z, x_mean, dis_real, dis_fake_fresh, lth_layer_real, lth_layer_recon,x_fresh

 
    def compute_loss(self, z_mean, z_log_sigma2, z, x_mean, dis_real, dis_fake_fresh, lth_layer_real, lth_layer_recon, y):
        """
        Loss functions for SSE, KL divergence, Discrim, Generator, Lth Layer Similarity
        """
        ### We don't actually use SSE (MSE) loss for anything (but maybe pretraining)
        # SSE_loss = tf.reduce_mean(tf.square(x - x_tilde)) # This is what a normal VAE uses

        dec_loss =  tf.reduce_sum(tf.square(lth_layer_real - lth_layer_recon)) / (1.0 * self.x_dim)

        # Discriminator Loss 
        label_extra_real = tf.concat(1,[tf.to_int32(y),tf.tile([[0]],[self.batch_size,1])])
        label_extra_fresh = tf.concat(1,[tf.tile([[0]],[self.batch_size,self.cond_info_dim]),tf.tile([[1]],[self.batch_size,1])]) 
               
        cross_entropy_real = tf.nn.softmax_cross_entropy_with_logits(dis_real, 
            label_extra_real, name='xentropy_real')

        cross_entropy_fresh = tf.nn.softmax_cross_entropy_with_logits(dis_fake_fresh, 
            label_extra_fresh, name='xentropy_fresh')

        dis_loss = tf.reduce_mean(cross_entropy_real + cross_entropy_fresh, name='xentropy_mean')

   
        enc_loss = tf.reduce_sum(-0.5 * tf.reduce_sum(1 + z_log_sigma2
                                           - tf.square(z_mean)
                                           - tf.exp(z_log_sigma2), 1)) / (1.0 * self.x_dim)
        # Lth Layer Loss - the 'learned similarity measure'  

        cross_entropy_fresh_fake = tf.nn.softmax_cross_entropy_with_logits(dis_fake_fresh, label_extra_real, name='xentropy_fresh_fake')
        g_loss = tf.reduce_mean(cross_entropy_fresh_fake)/ (1.0 * self.x_dim)
        prob_fake = tf.exp(-cross_entropy_fresh_fake)
        prob_real = tf.exp(-cross_entropy_real)
        return dec_loss,dis_loss,enc_loss,g_loss,prob_fake,prob_real

    def build_vae(self):
        """
        This function builds up VAE from encoder and decoder function
        in the global environment.
        """
        self.train_phase = tf.placeholder(tf.bool, name='train_phase')
        # Add a placeholder for one batch of images

        self.x = tf.placeholder(tf.float32,[self.batch_size,self.x_dim], name = 'x')

        self.y = tf.placeholder(tf.float32,[self.batch_size, self.cond_info_dim], name='y')


        z_mean, z_log_sigma2, z, x_mean, dis_real, dis_fake_fresh, lth_layer_real, lth_layer_recon,self.x_fresh = self.inference(self.x,self.y)
        dec_loss,dis_loss,enc_loss,g_loss,self.prob_fake,self.prob_real = self.compute_loss(z_mean, z_log_sigma2, z, x_mean, dis_real, dis_fake_fresh, lth_layer_real, lth_layer_recon, self.y)
         
        # Calculate the losses specific to encoder, generator, decoder
        self.L_d = tf.clip_by_value(dis_loss, -100, 100)
        self.L_g = tf.clip_by_value(dec_loss*self.beta+g_loss*self.gamma, -100, 100)
        self.L_e = tf.clip_by_value(enc_loss*self.alpha + dec_loss, -100, 100)

        # Add up to the cost.
        self.lr_D = tf.placeholder(tf.float32, shape=[])
        self.lr_G = tf.placeholder(tf.float32, shape=[])
        self.lr_E = tf.placeholder(tf.float32, shape=[])
        # Build a graph that trains the model with one batch of examples
        # and updates the model parameters.


        # specify loss to parameters
        params = tf.trainable_variables()
        E_params = [i for i in params if 'E' in i.name]
        G_params = [i for i in params if 'G' in i.name]
        D_params = [i for i in params if 'D' in i.name]


        self.d_opt = tf.train.AdamOptimizer(self.lr_D).minimize(self.L_d, var_list=D_params)
        self.g_opt = tf.train.AdamOptimizer(self.lr_G).minimize(self.L_g, var_list=G_params)
        self.e_opt = tf.train.AdamOptimizer(self.lr_E).minimize(self.L_e, var_list=E_params)

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

        return self.sess.run(self.x_fresh,\
                      feed_dict = {self.y:info, self.train_phase: False})

    def get_code(self, img):
        return self.sess.run(self.z_mean, feed_dict={self.x:img})

