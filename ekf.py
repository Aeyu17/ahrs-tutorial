from quaternion import Quaternion, ndarrayToQuat, skew3
import numpy as np 

from abc import ABC, abstractmethod

class EKF(ABC):
    def __init__(self, gyro_noise, accel_noise, freq, initial_state: Quaternion = Quaternion(1,0,0,0), initial_cov: np.ndarray = np.eye(4), gravity=9.81):
        # gyro measurement covariance
        self.Q = np.eye(3) * gyro_noise ** 2
        # accel measurement covariance
        self.R = np.eye(3) * accel_noise ** 2

        self.dt = 1/freq

        self.state: Quaternion = initial_state
        self.cov: np.ndarray = initial_cov
        self.g = gravity

    @abstractmethod
    def update(self, gyro: np.ndarray, accel: np.ndarray) -> tuple[Quaternion, np.ndarray]:
        pass

class FullStateEKF(EKF):
    def update(self, gyro: np.ndarray, accel: np.ndarray) -> tuple[Quaternion, np.ndarray]:
        if gyro.ndim != 1 or gyro.shape[0] != 3:
            raise TypeError('Invalid gyro measurement')
        
        if accel.ndim != 1 or accel.shape[0] != 3:
            raise TypeError('Invalid accel measurement')
        
        # NOTE: conventions followed from https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf and implemented using Kok et al.

        # -- Time Update --
        # x_k~ = f(x_k-1, u_k-1, 0)
        # this is specifically called imu mechanisation
        trans_gyro = 0.5 * self.dt * gyro
        gyro_mech_q = Quaternion(1, trans_gyro[0], trans_gyro[1], trans_gyro[2])
        q = (self.state @ gyro_mech_q).normalise()

        # P_k~ = A_k @ P_k-1 @ A_k' + W_k @ Q_k-1 @ W_k'
        A = gyro_mech_q.toRightMulMatrix()
        W = -0.5 * self.dt * self.state.toLeftMulMatrix()[:, 1:]
        P = A @ self.cov @ A.transpose() + W @ self.Q @ W.transpose()

        # -- Measurement Update --
        # K_k = P_k~ @ H_k' @ (H_k @ P_k~ @ H_k' + V_k @ R_k @ V_k')^-1
        H = 2 * self.g * np.matrix([[-q[2],  q[3], -q[0],  q[1]], 
                                    [ q[1],  q[0],  q[3],  q[2]], 
                                    [ q[0], -q[1], -q[2],  q[3]]])
        V = np.eye(3)
        K = P @ H.transpose() @ np.linalg.inv(H @ P @ H.transpose() + V @ self.R @ V.transpose())

        # x_k = x_k~ + K_k @ (z_k - h(x_k~, 0))
        h = q.toDCM().transpose() @ np.matrix([[0, 0, self.g]]).transpose()
        dq: np.ndarray = K @ (accel - h.transpose()).transpose()

        q = (q + Quaternion(dq[0, 0], dq[1, 0], dq[2, 0], dq[3, 0])).normalise()
        
        # P_k = (I - K_k @ H_k) @ P_k~
        P = (np.eye(4) - K @ H) @ P

        self.state = q
        self.cov = P

        return (q, P)
    
class ErrorStateEKF(EKF):
    def __init__(self, gyro_noise, accel_noise, freq, initial_state: Quaternion = Quaternion(1,0,0,0), initial_cov: np.ndarray = np.eye(3), gravity=9.81):
        super().__init__(gyro_noise=gyro_noise, accel_noise=accel_noise, freq=freq, initial_state=initial_state, initial_cov=initial_cov, gravity=gravity)   

    def update(self, gyro: np.ndarray, accel: np.ndarray) -> tuple[Quaternion, np.ndarray]:
        # -- Time Update --
        # x_k~ = f(x_k-1, u_k-1, 0)
        # this is specifically called imu mechanisation
        trans_gyro = 0.5 * self.dt * gyro
        gyro_mech_q = Quaternion(1, trans_gyro[0], trans_gyro[1], trans_gyro[2])
        q = (self.state @ gyro_mech_q).normalise()

        # P_k~ = A_k @ P_k-1 @ A_k' + W_k @ Q_k-1 @ W_k'
        A = np.eye(3)
        W = q.toDCM() * self.dt
        P = A @ self.cov @ A.transpose() + W @ self.Q @ W.transpose()

        # -- Measurement Update --
        # K_k = P_k~ @ H_k' @ (H_k @ P_k~ @ H_k' + V_k @ R_k @ V_k')^-1
        H = q.toDCM().transpose() @ skew3(np.array([0, 0, self.g]))
        V = np.eye(3)
        K = P @ H.transpose() @ np.linalg.inv(H @ P @ H.transpose() + V @ self.R @ V.transpose())

        # x_k = x_k~ + K_k @ (z_k - h(x_k~, 0))
        h = q.toDCM().transpose() @ np.matrix([[0, 0, self.g]]).transpose()
        dq: np.ndarray = K @ (accel - h.transpose()).transpose()
        q = (Quaternion(1, dq[0, 0], dq[1, 0], dq[2, 0]) @ q).normalise()
        
        # P_k = (I - K_k @ H_k) @ P_k~
        P = (np.eye(3) - K @ H) @ P

        self.state = q
        self.cov = P

        return (q, P)
    
if __name__ == '__main__':
    from import_h5 import importADPM, SensorData
    from quaternion import quatToEuler, incAccel
    import matplotlib.pyplot as plt

    filename = 'howard_arm_cal_646.h5'
    id = 'SI-000646'
    data: SensorData = importADPM(filename, id)

    n = data.gyro.shape[0]

    euler = quatToEuler(data.quat) # cba to convert them all into quaternion classes, not worth imo

    initial_q = ndarrayToQuat(incAccel(data.accel[0, :]))

    fs_ekf_euler: np.ndarray = np.zeros(shape=(n, 3))
    es_ekf_euler: np.ndarray = np.zeros(shape=(n, 3))

    fs_ekf = FullStateEKF(gyro_noise=0.001, accel_noise=0.1, freq=data.freq, initial_state=initial_q)
    es_ekf = ErrorStateEKF(gyro_noise=0.001, accel_noise=0.1, freq=data.freq, initial_state=initial_q)

    for i in range(n):
        q0, _ = fs_ekf.update(data.gyro[i, :], data.accel[i, :])
        fs_ekf_euler[i, :] = q0.toEuler()
        q1, _ = es_ekf.update(data.gyro[i, :], data.accel[i, :])
        es_ekf_euler[i, :] = q1.toEuler()

    for i in range(3):
        plt.figure()
        plt.plot(euler[:, i])
        plt.plot(fs_ekf_euler[:, i])
        plt.plot(es_ekf_euler[:, i])
        plt.legend(['acc', 'full state ekf', 'error state ekf'])
        plt.title(['Roll', 'Pitch', 'Yaw'][i])
    plt.show()
    

    