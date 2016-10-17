
import numpy as np

def get_middle_column(x):
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
    col = x[:, xrange(width/2, num_pixels+width/2, width)]
    out = np.random.binomial(1,col)
    return(out)
