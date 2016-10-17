
import numpy as np

def image2center(x):
    """
    x is an image (in a one-dimensional numpy array)
    we take the center column of the image,
    and sample binary values with probability equal
    to the corresponding pixel value
    """
    shape = x.shape

    if (len(shape) == 1):
        num_pixels = shape[0]
    else:
        num_pixels = shape[1]
    width = int(np.sqrt(num_pixels))
    col = x[xrange(width/2, num_pixels+width/2, width)]
    out = np.random.binomial(1,col)
    return(out)

