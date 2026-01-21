from import_h5 import importADPM, SensorData
from quaternion import *

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse as sps

def main():
    filename = 'howard_arm_cal_646.h5'
    id = 'SI-000646'
    data: SensorData = importADPM(filename, id)
    euler = quatToEuler(data.quat)

    # Full State EKF
    n = 10000
    dt = 1/data.freq

    # Quaternions are column vectors
    x = np.zeros(shape=(4, n+1))
    x[0, :] = np.ones(shape=(1, n+1))

    init_orientation = incAccel(data.accel[0, :]) # qv^bn, this is the prior after taking the conj

    g = 9.81

    import sys
    np.set_printoptions(threshold=sys.maxsize)

    for _ in range(10):

        # stack error
        e = np.zeros(shape=(6*n+3, 1)) # 3n gyro measurements + 3n accel measurements + 3 for initial error

        # initial error
        e[0:3, 0] = 2 * quatNormalise(quatMultiply(x[:, 0], quatConj(init_orientation)))[1:]

        # gyro error
        for i in range(n):
            e[3*i+3:3*i+6, 0] = (2 / dt) * quatNormalise(quatMultiply(quatConj(x[:, i]), x[:, i+1]))[1:] - data.gyro[i, :]

        # accel error
        for i in range(n):
            e[3*i + (3*n+3):3*i+3 + (3*n+3), 0] = data.accel[i+1, :] - quatToDCM(x[:, i+1]).transpose() @ np.array([0, 0, g])

        # jacobian
        j = sps.dok_matrix(np.zeros(shape=(6*n+3, 3*n+3)))
        # for jacobian reference, it is laid out like this:
        """
        a
        c b
          c b
            c b
              ...
          d
            d
              d
                ...
        """
        # where a is the jacobian of the initial e aka the prior, c is the jacobian of the current e w.r.t the gyro, b is the jacobian of the next w.r.t the gyro,
        # and d is the jacobian w.r.t the accel
        # in Kok et al., a is eq. 4.14a, b is eq. 4.14b, c is eq. 4.14c, and d is eq. 4.14d
        # could add mag below this using 4.14e

        # initial state jacobian
        jacobian_eta = np.zeros(shape=(4,4))

        q = quatNormalise(quatMultiply(x[:, 0], quatConj(init_orientation)))
        jacobian_eta[0, 0] = q[0]
        jacobian_eta[0, 1:] = -q[1:]
        jacobian_eta[1:, 0] = q[1:]
        jacobian_eta[1:, 1:] = q[0] * np.eye(3) - skew3(q[1:])

        weird_identity_matrix = np.zeros(shape=(4,3))
        weird_identity_matrix[1:, :] = np.eye(3)

        j[0:3, 0:3] = weird_identity_matrix.transpose() @ jacobian_eta @ weird_identity_matrix

        # gyro jacobian
        for i in range(n):
            q_bn = quatConj(x[:, i])
            q_bn_L = np.zeros(shape=(4,4))
            q_bn_L[0, 0] = q_bn[0]
            q_bn_L[0, 1:] = -q_bn[1:]
            q_bn_L[1:, 0] = q_bn[1:]
            q_bn_L[1:, 1:] = q_bn[0] * np.eye(3) + skew3(q_bn[1:])

            q_nb = x[:, i+1]
            q_nb_R = np.zeros(shape=(4,4))
            q_nb_R[0, 0] = q_nb[0]
            q_nb_R[0, 1:] = -q_nb[1:]
            q_nb_R[1:, 0] = q_nb[1:]
            q_nb_R[1:, 1:] = q_nb[0] * np.eye(3) - skew3(q_nb[1:])

            conj_derivative = np.eye(4)
            conj_derivative[1:, 1:] *= -1

            j[3*i+3:3*i+6, 3*i:3*i+3] = 1 / dt * weird_identity_matrix.transpose() @ q_bn_L @ q_nb_R @ conj_derivative @ weird_identity_matrix  # 4.14c
            j[3*i+3:3*i+6, 3*i+3:3*i+6] = 1 / dt * weird_identity_matrix.transpose() @ q_bn_L @ q_nb_R @ weird_identity_matrix # 4.14b

        # accel jacobian
        for i in range(n):
            j[3*i+3 + (3*n):3*i+6 + (3*n), 3*i+3:3*i+6] = -quatToDCM(x[:, i+1]).transpose() @ skew3(np.array([0, 0, g]))

        j = sps.csc_matrix(j) # convert from dok to sparse csc matrix, done because csc does not support item assignment

        # gradient
        # ((Jacobian * W_w * Jacobian') ** -1) * Jacobian * W_w * e = Jacobian * (W_w ** 1/2) * e = gradient
        # Assue W_w and W_a are identity for now; therefore gradient is just Jacobian * e
        G = j.transpose() @ e

        # hessian
        H = sps.csc_matrix(j.transpose() @ j)

        eta = -sps.linalg.inv(H) @ G

        exp_eta = np.ones(shape=(4, n+1))
        for i in range(n+1):
            exp_eta[1:, i] = eta[3*i:3*i+3, 0]/2
            x[:, i] = quatNormalise(quatMultiply(exp_eta[:, i], x[:, i]))

        print(e.transpose() @ e)

    # cov = np.linalg.inv(H) # uncomment if covariance is needed for some reason

    mechanised_q = np.zeros(shape=(4,n+1))
    mechanised_q[0, :] = np.ones(shape=(1,n+1))
    mechanised_q[:, 0] = init_orientation

    for i in range(n):
        mechanised_q[:, i+1] = quatNormalise(quatMultiply(mechanised_q[:, i], np.array([1, 0.5 * dt * data.gyro[i, 0], 0.5 * dt * data.gyro[i, 1], 0.5 * dt * data.gyro[i, 2]])))

    graph_opt_euler = quatToEuler(x.transpose())
    mechanised_euler = quatToEuler(mechanised_q.transpose())
    for i in range(3):
        plt.figure()
        plt.plot(mechanised_euler[:n, i], '-')
        plt.plot(graph_opt_euler[:n, i], '-')
        plt.plot(euler[:n, i], '-')
        plt.legend(['mech', 'graph opt', 'euler'])
        plt.title(['roll', 'pitch', 'yaw'][i])
    plt.show()

if __name__ == "__main__":
    main()
