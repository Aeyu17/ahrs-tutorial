from import_h5 import importADPM, SensorData
from quaternion import quatToEuler, quatNormalise, quatToDCM, incAccel
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
    # NOTE: (*) in comments means quat multiply
    x = np.zeros(shape=(4, n))
    x[0, :] = np.ones(shape=(1, n))

    x[:, 0] = incAccel(data.accel[0, :])

    P = np.eye(4)
    Q = np.eye(3) * gyr_noise ** 2
    R = np.eye(3) * acc_noise ** 2
    g = 9.81

    for i in range(n-1):
        # Time update
        f = np.zeros(shape=(4,4))
        f[0, 1:] = - data.gyro[i, :] # technically transposed, but not needed here
        f[1:, 0] = data.gyro[i, :]
        f[1:, 1:] = -skew3(data.gyro[i, :])
        f *= 0.5 * dt
        f += np.identity(4)

        x[:, i+1] = quatNormalise(f @ x[:, i])

        # P = A*P_(k-1)*A' + W*Q_(k-1)*W'
        A = f # why is this true? exercise left to reader, but it's in the Kok paper
        W = np.zeros(shape=(4,3))
        W[0, :] = - x[1:, i] # also technically transposed, but not needed
        W[1:, :] = x[0, i] * np.eye(3) + skew3(x[1:, i])
        P = A @ P @ A.transpose() + W @ Q @ W.transpose()

        # Measurement update
        # K = P*H' * (H*P*H' + V*R*V')^-1
        # H is the jacobian of the gravity vector rotated to the body frame w.r.t. the reference quaternion

        # H = [-2*g*xx(3,i+1),  2*g*xx(4,i+1), -2*g*xx(1,i+1), 2*g*xx(2,i+1);...
        #       2*g*xx(2,i+1),  2*g*xx(1,i+1),  2*g*xx(4,i+1), 2*g*xx(3,i+1);...
        #       2*g*xx(1,i+1), -2*g*xx(2,i+1), -2*g*xx(3,i+1), 2*g*xx(4,i+1)];

        H = np.array([[-x[2, i+1],  x[3, i+1], -x[0, i+1],  x[1, i+1]], 
                      [ x[1, i+1],  x[0, i+1],  x[3, i+1],  x[2, i+1]], 
                      [ x[0, i+1], -x[1, i+1], -x[2, i+1],  x[3, i+1]]])
        H *= 2 * g
        K = P @ H.transpose() @ np.linalg.inv(H @ P @ H.transpose() + R)

        # x = x - K * (z - h)
        h = quatToDCM(x[:, i+1]).transpose() @ np.array([0, 0, g])
        if i < 5:
            print(h)
        x[:, i+1] = x[:, i+1] + K @ (data.accel[i, :] - h)
        x[:, i+1] = quatNormalise(x[:, i+1])
        # P = (I - K*H) * P
        P = (np.eye(4) - K @ H) @ P

    ekf_euler = quatToEuler(x.transpose())
    plt.figure()
    plt.plot(euler[:, 1])
    plt.plot(ekf_euler[:, 1])
    # plt.plot(howard_euler[:, 1])
    plt.legend(['acc', 'kf', 'howard'])
    plt.show()

if __name__ == "__main__":
    main()
