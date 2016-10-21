import sys
from conv_deconv import Encoder, Decoder
from vae import Variational_Autoencoder
from ops import *    
import numpy as np
import tensorflow as tf
import argparse

 
def run_experiment(learning_rate, filter_dim, hidden_dim,num_epochs = 5):
    """
    This function runs the experiment and saves a checkpoint file with given
    learning_rate, filter_dim, hidden_dim,num_epochs.
    """
    checkpoint_name = 'vae_checkpoint_lr_{}_filter_dim_{}_hidden_dim_{}'.format( \
            learning_rate, filter_dim, hidden_dim)

    sess = tf.Session()
    build_encoder = Encoder(filter_dim, hidden_dim)
    build_decoder = Decoder(filter_dim)

    model = Variational_Autoencoder(sess,build_encoder, build_decoder, \
        checkpoint_name, batch_size = 100,z_dim = 20,
        img_dim = 784,dataset = 'mnist',
        learning_rate = learning_rate, num_epochs = num_epochs,
        checkpoint_dir = './')

    model.train()

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('learning_rate',type = float)
    parser.add_argument('filter_dim',type = int)
    parser.add_argument('hidden_dim',type = int)
    parser.add_argument('num_epochs',type = int)
    args = parser.parse_args()
    run_experiment(args.learning_rate,
        args.filter_dim, args.hidden_dim,
        args.num_epochs)

if __name__ == '__main__':
    main()
