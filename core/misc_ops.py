
import numpy as np
from PIL import Image

def get_image_subset(x, idx, sample=True):
    """
    x is an image (in a one-dimensional numpy array)
    we take the subset of pixels indexed by idx
    and if sample is true,
    sample binary values with probability equal
    to the corresponding pixel value,
    otherwise return the actual pixels
    """
    if (len(x.shape) == 1):
        x = x.reshape(1, len(x))
    subset = x[:, idx]
    if not sample:
        return(subset)
    out = np.random.binomial(1, subset)
    return(out)

def get_middle_column(x, sample=True):
    """
    x is an image (in a one-dimensional numpy array)
    we take the center column of the image,
    and sample binary values with probability equal
    to the corresponding pixel value
    """

    if (len(x.shape) == 1):
        x = x.reshape(1, len(x))
    num_pixels = x.shape[1]
    width = int(np.sqrt(num_pixels))
    #col = x[:, xrange(width/2, num_pixels+width/2, width)]
    #out = np.random.binomial(1, col)
    return(get_image_subset(x, xrange(width/2, num_pixels+width/2, width), sample=sample))
    #return(out)

def get_left_half(x, sample=True):
    if (len(x.shape) == 1):
        x = x.reshape(1, len(x))
    n = x.shape[0]
    num_pixels = x.shape[1]
    width = int(np.sqrt(num_pixels))
    subset = x.reshape(n, width, width)
    subset = subset[:, :, xrange(width/2)]
    subset = subset.reshape(n, width * width / 2)
    if not sample:
        return(subset)
    out = np.random.binomial(1, subset)
    return(out)

def gray_to_rgb(img):
    """
    Converts 28 x 28 array to 3 x 64 x 64 RGB
    """
    img = img.reshape(28,28)
    img = np.stack((img,img,img), axis=-1) # convert to RGB
    img = Image.fromarray(np.uint8(img*255))
    img = img.resize((64,64), Image.ANTIALIAS)
    img = np.asarray(img)
    img = np.rollaxis(img,2,0)
    return(img)

def get_feats(x, mean_img, model, layer):
    N = x.shape[0]
    x_rgb = np.zeros((N, 3, 64, 64))
    for i in xrange(N):
        x_rgb[i,] = gray_to_rgb(x[i,] - mean_img)
    feats, _ = model.forward(x_rgb, end=layer)
    feats = feats.reshape((N, -1))
    return(feats)
