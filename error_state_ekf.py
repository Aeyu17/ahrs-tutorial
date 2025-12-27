from import_h5 import importADPM, SensorData
from quaternion import quatToEuler, quatNormalise, quatToDCM, incAccel, quatMultiply
from matrix import skew3

import numpy as np
import matplotlib.pyplot as plt

def main():
    filename = 'howard_arm_cal_646.h5'
    id = 'SI-000646'
    data: SensorData = importADPM(filename, id)
    # howard_quat = np.loadtxt('bruh2.csv', delimiter=',')

    # howard_euler = quatToEuler(howard_quat)
    euler = quatToEuler(data.quat)

    # Full State EKF
    n = np.shape(data.gyro)[0]
    dt = 1/data.freq

    gyr_noise = 0.001
    acc_noise = 0.1

    # Quaternions are column vectors
    x = np.zeros(shape=(4, n))
    x[0, :] = np.ones(shape=(1, n))

    x[:, 0] = incAccel(data.accel[0, :])

    P = np.eye(3)
    Q = np.eye(3) * gyr_noise ** 2
    R = np.eye(3) * acc_noise ** 2
    g = 9.81

    for i in range(n-1):
        # Time update
        f = np.ones(shape=(4))
        f[1:] = 0.5 * dt * data.gyro[i, :]
        x[:, i+1] = quatNormalise(quatMultiply(x[:, i], f))
        # P = A*P_(k-1)*A' + W*Q_(k-1)*W'

        W = quatToDCM(x[:, i+1]) * dt
        P = P + W @ Q @ W.transpose()
        # Measurement update
        # K = P*H' * (H*P*H' + V*R*V')^-1
        C = quatToDCM(x[:, i+1])
        H = C.transpose() @ skew3(np.array([0, 0, g]))
        K = P @ H.transpose() @ np.linalg.inv(H @ P @ H.transpose() + R)

        # x = x + K * (z - h)
        h = C.transpose() @ np.array([0, 0, g])
        # x[:, i+1] = quatNormalise(x[:, i+1] + K @ (data.accel[i, :] - h))
        n = np.ones(shape=(4))
        n[1:] = K @ (data.accel[i, :] - h).transpose()
        x[:, i+1] = quatNormalise(quatMultiply(n, x[:, i+1]))

        # P = (I - K*H) * P
        P = (np.eye(3) - K @ H) @ P

    ekf_euler = quatToEuler(x.transpose())
    plt.figure()
    plt.plot(euler[:, 1])
    plt.plot(ekf_euler[:, 1])
    # plt.plot(howard_euler[:, 1])
    plt.legend(['acc', 'kf', 'howard'])
    plt.show()

if __name__ == "__main__":
    main()
