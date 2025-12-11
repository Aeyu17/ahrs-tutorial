import numpy as np
from math import atan2, asin, sqrt

def quatToEuler(quat: np.ndarray):
    if quat.ndim != 2 or np.shape(quat)[1] != 4:
        raise ValueError('Invalid quaternion matrix')
    
    eulerArr = np.zeros(shape=(np.shape(quat)[0], 3))
    
    for i, q in enumerate(quat):
        eulerArr[i, :] = singleQuatToEuler(q)

    return eulerArr

def singleQuatToEuler(q: np.ndarray):
    if q.ndim != 1 or np.shape(q)[0] != 4:
        raise ValueError('Invalid quaternion matrix')
    
    roll = atan2(2 * (q[0] * q[1] + q[2] * q[3]), q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2)
    pitch = asin(2 * (q[0] * q[2] - q[3] * q[1]))
    yaw = atan2(2 * (q[0] * q[3] + q[1] * q[2]), q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2)

    return roll, pitch, yaw

def quatNormalise(q: np.ndarray):
    if q.ndim != 1 or np.shape(q)[0] != 4:
        raise ValueError('Invalid quaternion matrix')
    norm = sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    return q / norm