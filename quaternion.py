import numpy as np
from math import atan2, asin, sqrt, cos, sin

def quatToEuler(quat: np.ndarray):
    # Asking for 2 dim quaternion because of time series
    if quat.ndim != 2 or np.shape(quat)[1] != 4:
        raise ValueError('Invalid quaternion matrix')
    
    eulerArr = np.zeros(shape=(np.shape(quat)[0], 3))
    
    for i, q in enumerate(quat):
        eulerArr[i, :] = singleQuatToEuler(q)

    return eulerArr

def quatMultiply(q1: np.ndarray, q2: np.ndarray):
    if q1.ndim != 1 or np.shape(q1)[0] != 4:
        print(q1)
        raise ValueError('q1 is an invalid quaternion matrix')
    
    if q2.ndim != 1 or np.shape(q2)[0] != 4:
        raise ValueError('q2 is an invalid quaternion matrix')
    
    q = np.array([q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
                  q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
                  q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
                  q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]])
    
    return q

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

def skew3(vec):
    if len(vec) != 3:
        raise ValueError('Skew function only takes vectors of size 3')
    return np.cross(vec, np.identity(vec.shape[0]) * -1)

def ndarrayToQuat(q: np.ndarray) -> Quaternion:
    if q.ndim != 1 or np.shape(q)[0] != 4:
        raise ValueError('Invalid quaternion matrix')
    
    return Quaternion(q[0], q[1], q[2], q[3])

class Quaternion:
    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.quat = np.array([w, x, y, z])

    def w(self) -> float:
        # returns the real part of q
        return self.quat[0]
    
    def vector(self) -> np.ndarray:
        # returns the imaginary part of q
        return self.quat[1:]

    def __getitem__(self, val):
        return self.quat[val]

    def __matmul__(self, other) -> Quaternion:
        # using __matmul__ to define quaternion multiplication
        if not isinstance(other, Quaternion):
            raise NotImplemented
        
        q = Quaternion(self[0] * other[0] - self[1] * other[1] - self[2] * other[2] - self[3] * other[3],
                       self[0] * other[1] + self[1] * other[0] + self[2] * other[3] - self[3] * other[2],
                       self[0] * other[2] - self[1] * other[3] + self[2] * other[0] + self[3] * other[1],
                       self[0] * other[3] + self[1] * other[2] - self[2] * other[1] + self[3] * other[0])
        
        return q
    
    def toEuler(self) -> tuple[float, float, float]:
        roll = atan2(2 * (self[0] * self[1] + self[2] * self[3]), self[0]**2 - self[1]**2 - self[2]**2 + self[3]**2)
        pitch = asin(2 * (self[0] * self[2] - self[3] * self[1]))
        yaw = atan2(2 * (self[0] * self[3] + self[1] * self[2]), self[0]**2 + self[1]**2 - self[2]**2 - self[3]**2)

        return (roll, pitch, yaw)
    
    def normalise(self) -> Quaternion:
        # returns a normalised quaternion made of itself
        norm = sqrt(self[0]**2 + self[1]**2 + self[2]**2 + self[3]**2)
        return Quaternion(self.quat[0] / norm, self.quat[1] / norm, self.quat[2] / norm, self.quat[3] / norm)
    
    def conj(self) -> Quaternion:
        return Quaternion(self.quat[0], -self.quat[1], -self.quat[2], -self.quat[3])
        
    def toDCM(self) -> np.ndarray:
        dcm = np.zeros(shape=(3,3))
        
        dcm[0, 0] = self[0] ** 2 + self[1] ** 2 - self[2] ** 2 - self[3] ** 2
        dcm[0, 1] = -2 * (self[0] * self[3] - self[1] * self[2])
        dcm[0, 2] =  2 * (self[0] * self[2] + self[1] * self[3])

        dcm[1, 0] =  2 * (self[0] * self[3] + self[1] * self[2])
        dcm[1, 1] = self[0] ** 2 - self[1] ** 2 + self[2] ** 2 - self[3] ** 2
        dcm[1, 2] = -2 * (self[0] * self[1] - self[2] * self[3])

        dcm[2, 0] = -2 * (self[0] * self[2] - self[1] * self[3])
        dcm[2, 1] =  2 * (self[0] * self[1] + self[2] * self[3])
        dcm[2, 2] = self[0] ** 2 - self[1] ** 2 - self[2] ** 2 + self[3] ** 2
        
        return dcm
    
    def toLeftMulMatrix(self) -> np.ndarray:
        leftMul = np.zeros(shape=(4,4))
        leftMul[0, 0] = self.w()
        leftMul[0, 1:] = -self.vector()
        leftMul[1:, 0] = self.vector()
        leftMul[1:, 1:] = self.w() * np.eye(3) + skew3(self.vector())
        return leftMul

    def toRightMulMatrix(self) -> np.ndarray:
        rightMul = np.zeros(shape=(4,4))
        rightMul[0, 0] = self.w()
        rightMul[0, 1:] = -self.vector()
        rightMul[1:, 0] = self.vector()
        rightMul[1:, 1:] = self.w() * np.eye(3) - skew3(self.vector())