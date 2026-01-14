from import_h5 import importADPM, SensorData
from quaternion import quatToEuler, quatNormalise, incAccel, quatMultiply, quatConj
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
    n = 1
    dt = 1/data.freq

    # Quaternions are column vectors
    x = np.zeros(shape=(4, n))
    x[0, :] = np.ones(shape=(1, n))
    # x[:, 0] # q^~nb

    init_orientation = incAccel(data.accel[0, :]) # qv^bn, this is the prior after taking the conj

    # gyr_noise = 0.001
    # acc_noise = 0.1

    # W_w = np.eye(3) * gyr_noise ** -1
    # W_a = np.eye(3) * acc_noise ** -1

    eta = np.zeros(shape=(3, 1))


    for k in range(10):

        # e = [e_n' e_w']'
        e = np.zeros(shape=(3)) # 3n gyro measurements + 3 for initial error

        # initial error
        e[0:3] = 2 * (quatMultiply(x[:, 0], quatConj(init_orientation)))[1:]

        # fill in the gyro...
        # for i in range(n):
        #     e[3*i+3:3*i+6] = 2 / dt * (quatMultiply(quatConj(x[:, i]), )) - data.gyro[i, :]


        # jacobian building
        j = np.zeros(shape=(3, 3))

        # initial jacobian
        jacobian_eta = np.zeros(shape=(4,4))

        q = quatMultiply(x[:, 0], quatConj(init_orientation))
        jacobian_eta[0, 0] = q[0]
        jacobian_eta[0, 1:] = -q[1:]
        jacobian_eta[1:, 0] = q[1:]
        jacobian_eta[1:, 1:] = q[0] * np.eye(3) - skew3(q[1:])

        weird_identity_matrix = np.zeros(shape=(4,3))
        weird_identity_matrix[1:, :] = np.eye(3)

        j[0:3, 0:3] = weird_identity_matrix.transpose() @ jacobian_eta @ weird_identity_matrix

        # fill in the gyro...

        # gradient
        # ((Jacobian * W_w * Jacobian') ** -1) * Jacobian * W_w * e = Jacobian * (W_w ** 1/2) * e = gradient
        # Assue W_w and W_a are identity for now; therefore gradient is just Jacobian * e
        G = j.transpose() @ e

        # hessian
        H = j.transpose() @ j

        eta = -np.linalg.inv(H) @ G
        # print(eta.shape)
        # print(eta)
        exp_eta = np.ones(shape=(4))
        exp_eta[1:] = eta/2
        exp_eta = quatNormalise(exp_eta)
        x[:, 0] = quatNormalise(quatMultiply(exp_eta[:], x[:, 0]))


        eta = np.zeros(shape=(3, n))


        print(e @ e.transpose())
    
    print(init_orientation)
    print(x)

    # graph_opt_euler = quatToEuler(x.transpose())
    # plt.figure()
    # plt.plot(euler[:n, 1])
    # plt.plot(graph_opt_euler[:n, 1])
    # # plt.plot(howard_euler[:, 1])
    # plt.legend(['acc', 'kf', 'howard'])
    # plt.show()

if __name__ == "__main__":
    main()
