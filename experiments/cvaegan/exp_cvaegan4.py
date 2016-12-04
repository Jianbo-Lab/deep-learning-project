import sys 
#from vaegan_fc import *
#from vaegan_conv import *


from misc_ops import *

import numpy as np
import tensorflow as tf
 
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets 
import argparse
from dataset import DataSet
import scipy.io as sio
from cvaegan_conv3 import *
from cvaegan4 import CVAEGAN
def mat2dataset(filepath):
    dat = sio.loadmat(filepath)
    x = dat['X']
    y = dat['y']
    y[y==10] = 0
    y = np.array(np.arange(10)==y, dtype=int)
    x = x.reshape((32*32*3,-1)).T
    x = x.astype(float)/255.
    return DataSet(x,y,dtype=dtypes.float32, reshape=False) 
 
def run_experiment(learning_rate, alpha, beta,Encoder,Decoder,Discriminator,gamma = 1,num_epochs = 5,dataset = 'mnist',version = 3,layer = 4):
	"""
	This function runs the experiment and saves a checkpoint file with given
	learning_rate, filter_dim, hidden_dim,num_epochs.
	""" 

	checkpoint_name = 'cvaegan_checkpoint_lr_{}_alpha_{}_beta_{}_gamma_{}_{}_net_{}_v4'.format( \
		        learning_rate, alpha, beta, gamma,dataset,version) 

	if dataset == 'mnist':
		# mnist = read_data_sets('MNIST_data', one_hot=True)
		
		sess = tf.InteractiveSession()
		build_encoder = Encoder(256)
		build_decoder = Decoder(256)
		build_discriminator = Discriminator(512)

		model = CVAEGAN(sess, build_encoder, build_decoder, build_discriminator, dataset='mnist',
		            batch_size = 100, z_dim = 50, x_dim = 784,learning_rate = learning_rate, num_epochs = num_epochs, 
		            load=False, lr_decay=0.95, alpha = alpha, beta = beta, gamma=gamma, checkpoint_name=checkpoint_name, 
		            checkpoint_dir = './')
		model.train()
	elif dataset == 'SVHN':
		svhn_train = mat2dataset('SVHN/train_32x32.mat')
		svhn_test = mat2dataset('SVHN/test_32x32.mat')

		sess = tf.InteractiveSession()
		build_encoder = Encoder(256, x_width = 32)
		if version == 4:
			layer = 5
		build_decoder = Decoder(256, x_width = 32,x_depth = 3,layer = layer)
		build_discriminator = Discriminator(512, x_width = 32)

		# model = CVAEGAN(sess, build_encoder, build_decoder, build_discriminator, 
		# 	dataset=svhn_train,testset = svhn_test,batch_size = 100, z_dim = 50, 
		# 	x_dim = 32*32*3,learning_rate = learning_rate, num_epochs = num_epochs, 
		# 	load=False, lr_decay=1., alpha = alpha, beta = beta, gamma=gamma, 
		# 	checkpoint_name=checkpoint_name,checkpoint_dir = './')
		model = CVAEGAN(sess, build_encoder, build_decoder, build_discriminator, 
			dataset=svhn_train,testset = svhn_test,batch_size = 100, z_dim = 50, 
			x_dim = 32*32*3,learning_rate = learning_rate, num_epochs = num_epochs, 
			load=True, load_file = 'sslgan_conv_checkpoint_0.0001_1.0_1.0_1.0_100-19',  lr_decay=1., alpha = alpha, beta = beta, gamma=gamma, 
			checkpoint_name=checkpoint_name,checkpoint_dir = './')

		model.train()		

def main(): 
	parser = argparse.ArgumentParser()
	parser.add_argument('learning_rate',type = float)
	parser.add_argument('alpha',type = float)
	parser.add_argument('beta',type = float)
	parser.add_argument('gamma',type = float)
	parser.add_argument('num_epochs',type = int)
	parser.add_argument('--dataset',type = str,default = 'mnist') 
	parser.add_argument('--version',type = int,default = 3) 
	args = parser.parse_args()
	if args.version == 3:
		run_experiment(args.learning_rate,
		args.alpha, args.beta, Encoder,Decoder,Discriminator, args.gamma,
		args.num_epochs,args.dataset)
	elif args.version == 4:
		run_experiment(args.learning_rate,
		args.alpha, args.beta, Encoder2,Decoder2,Discriminator2, args.gamma,
		args.num_epochs,args.dataset,args.version)		

if __name__ == '__main__':
    main()
