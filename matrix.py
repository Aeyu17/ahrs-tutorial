import numpy as np

def skew3(vec):
    if len(vec) != 3:
        raise ValueError('Skew function only takes vectors of size 3')
    return np.cross(vec, np.identity(vec.shape[0]) * -1)