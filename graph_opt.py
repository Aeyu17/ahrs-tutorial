from import_h5 import importADPM, SensorData
from quaternion import quatToEuler, quatNormalise, incAccel, quatMultiply, quatConj, singleQuatToEuler
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
    n = 10
    dt = 1/data.freq

    # Quaternions are column vectors
    x = np.zeros(shape=(4, n+1))
    x[0, :] = np.ones(shape=(1, n+1))
    # x[:, 0] # q^~nb

    init_orientation = incAccel(data.accel[0, :]) # qv^bn, this is the prior after taking the conj
    # init_orientation = np.array([1,0,0,0])

    # gyr_noise = 0.001
    # acc_noise = 0.1

    # W_w = np.eye(3) * gyr_noise ** -1
    # W_a = np.eye(3) * acc_noise ** -1

    import sys
    np.set_printoptions(threshold=sys.maxsize)

    for _ in range(100):

        # e = [e_n' e_w']'
        e = np.zeros(shape=(3*n+3, 1)) # 3n gyro measurements + 3 for initial error

        # initial error
        e[0:3, 0] = 2 * quatNormalise(quatMultiply(x[:, 0], quatConj(init_orientation)))[1:]

        # fill in the gyro...
        for i in range(n):
            e[3*i+3:3*i+6, 0] = (2 / dt) * quatNormalise(quatMultiply(quatConj(x[:, i]), x[:, i+1]))[1:] - data.gyro[i, :]

        # print(e)
        # print()

        # jacobian building
        j = np.zeros(shape=(3*n+3, 3*n+3))

        # initial jacobian
        jacobian_eta = np.zeros(shape=(4,4))

        q = quatNormalise(quatMultiply(x[:, 0], quatConj(init_orientation)))
        jacobian_eta[0, 0] = q[0]
        jacobian_eta[0, 1:] = -q[1:]
        jacobian_eta[1:, 0] = q[1:]
        jacobian_eta[1:, 1:] = q[0] * np.eye(3) - skew3(q[1:])

        weird_identity_matrix = np.zeros(shape=(4,3))
        weird_identity_matrix[1:, :] = np.eye(3)

        j[0:3, 0:3] = weird_identity_matrix.transpose() @ jacobian_eta @ weird_identity_matrix

        # fill in the gyro...
        for i in range(n):
            q_bn = quatConj(x[:, i])
            q_bn_L = np.zeros(shape=(4,4))
            q_bn_L[0, 0] = q_bn[0]
            q_bn_L[0, 1:] = -q_bn[1:]
            q_bn_L[1:, 0] = q_bn[1:]
            q_bn_L[1:, 1:] = q_bn[0] * np.eye(3) + skew3(q_bn[1:])

            # print('qbnl')
            # print(q_bn_L)
            # print()

            q_nb = x[:, i+1]
            q_nb_R = np.zeros(shape=(4,4))
            q_nb_R[0, 0] = q_nb[0]
            q_nb_R[0, 1:] = -q_nb[1:]
            q_nb_R[1:, 0] = q_nb[1:]
            q_nb_R[1:, 1:] = q_nb[0] * np.eye(3) - skew3(q_nb[1:])

            # print('qnbr')
            # print(q_nb_R)
            # print()

            conj_derivative = np.eye(4)
            conj_derivative[1:, 1:] *= -1

            # print('conj derivative')
            # print(conj_derivative)
            # print()

            # print('logq')
            # print(weird_identity_matrix.transpose())
            # print()

            j[3*i+3:3*i+6, 3*i:3*i+3] = 1 / dt * weird_identity_matrix.transpose() @ q_bn_L @ q_nb_R @ conj_derivative @ weird_identity_matrix  # 4.14c
            j[3*i+3:3*i+6, 3*i+3:3*i+6] = 1 / dt * weird_identity_matrix.transpose() @ q_bn_L @ q_nb_R @ weird_identity_matrix # 4.14b

        # print(j[-7:-1, -7:-1])

        # for i in range(3*n+3):
        #     str = ""
        #     for p in range(3*n+3):
        #         str += f'{j[i, p]:10,.2f},'
        #     print(str)

        # gradient
        # ((Jacobian * W_w * Jacobian') ** -1) * Jacobian * W_w * e = Jacobian * (W_w ** 1/2) * e = gradient
        # Assue W_w and W_a are identity for now; therefore gradient is just Jacobian * e
        G = j.transpose() @ e
        # print(G)
        # print()

        # hessian
        H = j.transpose() @ j
        # print(H)


        # eta = np.reshape(-np.linalg.inv(H) @ G, shape=(3, n))
        # exp_eta = np.ones(shape=(4, n))
        # exp_eta[1:, :] = eta
        # for i in range(n-1):
        #     print(eta[:, i])
        #     exp_eta[:, i] = quatNormalise(exp_eta[:, i])
        #     x[:, i] = quatNormalise(quatMultiply(exp_eta[:, i], x[:, i]))

        eta = -np.linalg.inv(H) @ G

        # print(eta.shape)

        exp_eta = np.ones(shape=(4, n+1))
        for i in range(n+1):
            # print(eta[3*i:3*i+3])
            # print(f'{3*i}, {3*i+3}')
            exp_eta[1:, i] = eta[3*i:3*i+3, 0]/2
            # exp_eta[:, i] = quatNormalise(exp_eta[:, i])
            x[:, i] = quatNormalise(quatMultiply(exp_eta[:, i], x[:, i]))

        # exp_eta = np.ones(shape=(4))
        # exp_eta[1:] = eta/2
        # exp_eta = quatNormalise(exp_eta)
        # x[:, 0] = quatNormalise(quatMultiply(exp_eta[:], x[:, 0]))

        # eta = np.zeros(shape=(3*n+3))

        # for i in range(n+1):
        #     print(x[:, i])
        print(e.transpose() @ e)
        # print(e)
        # print(e.shape)

    qppp = np.zeros(shape=(4,n+1))
    qppp[0, :] = np.ones(shape=(1,n+1))
    qppp[:, 0] = incAccel(data.accel[0, :])
    print(singleQuatToEuler(qppp[:, 0]))

    for i in range(10):
        qppp[:, i+1] = quatNormalise(quatMultiply(qppp[:, i], np.array([1, 0.5 * dt * data.gyro[i+1, 0], 0.5 * dt * data.gyro[i+1, 1], 0.5 * dt * data.gyro[i+1, 2]])))
        # print(q)
    

    graph_opt_euler = quatToEuler(x[:, 1:].transpose())
    mechanised_euler = quatToEuler(qppp.transpose())
    for i in range(3):
        plt.figure()
        plt.plot(mechanised_euler[:n-1, i])
        plt.plot(graph_opt_euler[:n-1, i])
        plt.legend(['mech', 'graph opt'])
    plt.show()

if __name__ == "__main__":
    main()
