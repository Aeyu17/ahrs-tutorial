from import_h5 import importADPM, SensorData
from quaternion import quatToEuler, quatNormalise
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

    for i in range(n-1):
        # Time update
        # x = f(x_(k-1), u_(k-1), 0) becomes...
        # q_t = q_t-1 (*) exp_q(dt/2 * y_w,t-1) per Kok paper 4.45a
        # exp_q = (cos |n|; n/|n| * sin|n|) ~= (1; n) where n is our gyro measurements
        # then you do quaternion righthand multiplication, hence...
        # def f = exp_q(dt/2 * y_w,t-1) = [1, -dt/2 * w'; dt/2 * w, I3 - skew(-dt/2 * w)];
        #       = I4 + dt/2 [0, -w'; w, -skew(w)];
        # then, because it's righthand mult, f * x_t-1

        f = np.zeros(shape=(4,4))
        f[0, 1:] = - data.gyro[i, :] # technically transposed, but not needed heres
        f[1:, 0] = data.gyro[i, :]
        f[1:, 1:] = -skew3(data.gyro[i, :])
        f *= 0.5 * dt
        f += np.identity(4)

        x[:, i+1] = quatNormalise(f @ x[:, i])


        # % P = A*P_(k-1)*A' + W*Q_(k-1)*W'

        # % Measurement update
        # % K = P*H' * (H*P*H' + V*R*V')^-1
        # % x = x - K * (z - h)
        # % P = (I - K*H) * P

    ekf_euler = quatToEuler(x.transpose())
    plt.figure()
    plt.plot(euler[:, 1])
    plt.plot(ekf_euler[:, 1])
    # plt.plot(howard_euler[:, 1])
    plt.legend(['acc', 'kf', 'howard'])
    plt.show()

if __name__ == "__main__":
    main()
