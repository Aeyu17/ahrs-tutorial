import numpy as np
from math import atan2, asin, sqrt, cos, sin

def quatToEuler(quat: np.ndarray):
    if quat.ndim != 2 or np.shape(quat)[1] != 4:
        raise ValueError('Invalid quaternion matrix')
    
    eulerArr = np.zeros(shape=(np.shape(quat)[0], 3))
    
    for i, q in enumerate(quat):
        eulerArr[i, :] = singleQuatToEuler(q)

    return eulerArr

def incAccel(accel):
    # %calculate pitch and roll from accel
    # pitch=atan2(-accel(:,1),sqrt(accel(:,2).^2+accel(:,3).^2)); %pitch
    # roll=atan2(accel(:,2),accel(:,3));  %roll
    pitch = atan2(-accel[0], sqrt(accel[1] ** 2 + accel[2] ** 2))
    roll = atan2(accel[1], accel[2])

    # %Convert to quaternion    
    # q(:,1)=cos(pitch./2).*cos(roll./2);
    # q(:,2)=cos(pitch./2).*sin(roll./2);
    # q(:,3)=sin(pitch./2).*cos(roll./2);
    # q(:,4)=-sin(pitch./2).*sin(roll./2);
    q = np.zeros(shape=(4))
    q[0] = cos(pitch / 2) * cos(roll / 2)
    q[1] = cos(pitch / 2) * sin(roll / 2)
    q[2] = sin(pitch / 2) * cos(roll / 2)
    q[3] = sin(pitch / 2) * sin(roll / 2)

    return q


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

def quatConj(q: np.ndarray):
    if q.ndim != 1 or np.shape(q)[0] != 4:
        raise ValueError('Invalid quaternion matrix')
    
    quat = np.ndarray(shape=(4))
    quat[0] = q[0]
    quat[1:] = -q[1:]
    return quat

def quatToDCM(q: np.ndarray):
    if q.ndim != 1 or np.shape(q)[0] != 4:
        raise ValueError('Invalid quaternion matrix')
    
    dcm = np.zeros(shape=(3,3))

        
    # R(1,1)= q(1)^2+q(2)^2-q(3)^2-q(4)^2;
    # R(1,2)=-2*q(1)*q(4)+2*q(2)*q(3);
    # R(1,3)= 2*q(1)*q(3)+2*q(2)*q(4);
    dcm[0, 0] = q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2
    dcm[0, 1] = -2 * (q[0] * q[3] - q[1] * q[2])
    dcm[0, 2] =  2 * (q[0] * q[2] + q[1] * q[3])

    # R(2,1)= 2*q(1)*q(4)+2*q(2)*q(3);
    # R(2,2)= q(1)^2-q(2)^2+q(3)^2-q(4)^2;
    # R(2,3)=-2*q(1)*q(2)+2*q(3)*q(4);
    dcm[1, 0] =  2 * (q[0] * q[3] + q[1] * q[2])
    dcm[1, 1] = q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2
    dcm[1, 2] = -2 * (q[0] * q[1] - q[2] * q[3])

    # R(3,1)=-2*q(1)*q(3)+2*q(2)*q(4);
    # R(3,2)=2*q(1)*q(2)+2*q(3)*q(4);
    # R(3,3)=q(1)^2-q(2)^2-q(3)^2+q(4)^2;
    dcm[2, 0] = -2 * (q[0] * q[2] - q[1] * q[3])
    dcm[2, 1] =  2 * (q[0] * q[1] + q[2] * q[3])
    dcm[2, 2] = q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2
    
    return dcm