
import numpy as np

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
