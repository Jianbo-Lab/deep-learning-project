import sys
sys.path.append('../core/')
sys.path.append('../networks/')
import time, os, json
import numpy as np
from scipy.misc import imread, imresize 

from cs294_129.classifiers.pretrained_cnn import PretrainedCNN
from cs294_129.data_utils import load_tiny_imagenet
from cs294_129.image_utils import blur_image, deprocess_image, preprocess_image
 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def invert_features(target_feats, layer, model, **kwargs):
  """
  Perform feature inversion in the style of Mahendran and Vedaldi 2015, using
  L2 regularization and periodic blurring.
  
  Inputs:
  - target_feats: Image features of the target image, of shape (1, C, H, W);
    we will try to generate an image that matches these features
  - layer: The index of the layer from which the features were extracted
  - model: A PretrainedCNN that was used to extract features
  
  Keyword arguments:
  - learning_rate: The learning rate to use for gradient descent
  - num_iterations: The number of iterations to use for gradient descent
  - l2_reg: The strength of L2 regularization to use; this is lambda in the
    equation above.
  - blur_every: How often to blur the image as implicit regularization; set
    to 0 to disable blurring.
  - show_every: How often to show the generated image; set to 0 to disable
    showing intermediate reuslts.
    
  Returns:
  - X: Generated image of shape (1, 3, 64, 64) that matches the target features.
  """
  learning_rate = kwargs.pop('learning_rate', 10000)
  num_iterations = kwargs.pop('num_iterations', 500)
  l2_reg = kwargs.pop('l2_reg', 1e-7)
  blur_every = kwargs.pop('blur_every', 1)
  show_every = kwargs.pop('show_every', 50)
  
  X = np.random.randn(1, 3, 64, 64)
  for t in xrange(num_iterations):
    ############################################################################
    # TODO: Compute the image gradient dX of the reconstruction loss with      #
    # respect to the image. You should include L2 regularization penalizing    #
    # large pixel values in the generated image using the l2_reg parameter;    #
    # then update the generated image using the learning_rate from above.      #
    ############################################################################
    feats_X,cache = model.forward(X,end = layer)
    dout = 2 * (feats_X - target_feats) 
    dX,grads = model.backward(dout,cache) 
    dX += 2 * l2_reg * X 
    X -= learning_rate * dX
#     if t % 100 == 0:
#         loss = np.sum((feats_X - target_feats) ** 2) + l2_reg * np.sum(X ** 2)
#         print 'iterations {}: loss {}.'.format(t,loss)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    # As a regularizer, clip the image
    X = np.clip(X, -data['mean_image'], 255.0 - data['mean_image'])
    
    # As a regularizer, periodically blur the image
    if (blur_every > 0) and t % blur_every == 0:
      X = blur_image(X)

    if (show_every > 0) and (t % show_every == 0 or t + 1 == num_iterations):
      plt.imshow(deprocess_image(X, data['mean_image']))
      plt.gcf().set_size_inches(3, 3)
      plt.axis('off')
      plt.title('t = %d' % t)
      plt.show()
 



def preprocess_image(imgs, mean_img):
  """
  Convert to float, transepose, and subtract mean pixel
  
  Input:
  - img: (N, H, W, 3)
  
  Returns:
  - (N, H, W, 3)
  """ 
  return imgs.astype(np.float32) - mean_img.astype(np.float32)


def image_to_feats(imgs,mean_img,model,layer):
  """
  This function transforms an image to features in the specified layer.
  (to be input into the encoder).
  Input: 
  imgs: Images to be transformed. N * H * W * 3.
  mean_img: mean to be subtracted from img.
  model: pretrained model to be used.
  layer: the number of layer to be specified.

  """
  # Preprocess the image before passing it to the network:
  # subtract the mean, add a dimension, etc
  imgs_pre = preprocess_image(imgs, mean_img)

  # Extract features from the image
  feats, _ = model.forward(imgs_pre, end=layer)

  return feats

def convert_images(batch_images):
  # Convert to the size suitable for the pretrained network.
  images = tf.image.resize_images(batch_images, 64, 64, \
    method=0, align_corners=False) 
  images = tf.image.convert_image_dtype(images, tf.float32)
  rgb_images = tf.image.grayscale_to_rgb(images, name=None)
  return rgb_images

def main():

  sess = tf.InteractiveSession()


  mnist = input_data.read_data_sets('MNIST_data', one_hot=True) 
  layer = 4
  batch_size = 100
  allimages = mnist.train.images.reshape(55000,28,28,1)
  mean_img = np.mean(allimages,axis = 0)
  mean_img = convert_images(mean_img).eval()
  mean_img = np.rollaxis(mean_img,2,0)
  
  for i in range(55000 / batch_size):
    batch_images = allimages[i*batch_size:(i+1)*batch_size]
     
    rgb_images = convert_images(batch_images)

    model = PretrainedCNN(h5_file='../networks/cs294_129/datasets/pretrained_model.h5')

    rgb_images = rgb_images.eval() 
    rgb_images = np.rollaxis(rgb_images,3,1) 

    feats = image_to_feats(rgb_images,mean_img,model,layer) 
    print feats.shape
if __name__ == '__main__':
  main()