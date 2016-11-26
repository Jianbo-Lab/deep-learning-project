import numpy as np
import tensorflow as tf
import time
import os

class SSL_M1M2():
    def __init__(self, sess, build_encoder1, build_encoder2, build_decoder, dataset_l, dataset_u,
        checkpoint_name = 'SSL_M2_checkpoint',
        batch_size = 100, z_dim = 20, x_dim = 784, y_dim = 10, alpha = 5500.,
        learning_rate = 0.001, num_epochs = 5,load = False,load_file = None,
        checkpoint_dir = '../notebook/checkpoints/', summaries_dir = 'm2_logs/'):
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

        self.n_samples_l = dataset_l.num_examples
        self.n_samples_u = dataset_u.num_examples
