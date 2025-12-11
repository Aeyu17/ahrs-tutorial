import numpy as np

def skew3(vec):
    if len(vec) != 3:
        raise ValueError('Skew function only takes vectors of size 3')
    return np.array([[      0, -vec[2],  vec[1]],
                     [ vec[2],       0, -vec[0]],
                     [-vec[1],  vec[0],       0]])