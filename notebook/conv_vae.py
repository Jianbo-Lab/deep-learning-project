import sys
sys.path.append('../core/')
sys.path.append('../networks/')
from conv_deconv import build_encoder, build_decoder
from vae import Variational_Autoencoder
    
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt 
def main():
	sess = tf.Session()
	model = Variational_Autoencoder(sess,build_encoder, build_decoder, batch_size = 100,z_dim = 20,img_dim = 784,dataset = 'mnist',
	                              learning_rate = 0.005, num_epochs = 5)
	model.train()
	

if name == '__main__':
	main()