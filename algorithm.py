from config import X_initial_state, H_Lidar,R_lidar, initialCovMatrix
from vizualization import *



def computeRadarJacobian(Xvector):
    px = Xvector[0][0]
    py = Xvector[1][0]
    vx = Xvector[2][0]
    vy = Xvector[3][0]
    sum_square = np.power(px, 2) + np.power(py, 2)
    two_thirds = sum_square ** 1.5
    dist = np.sqrt(sum_square)

    return np.array([[px / dist, py / dist, 0, 0],
                     [-px / sum_square, px / sum_square, 0, 0],
                     [(py * (vx * py - vy * px)) / two_thirds,
                      (px * (vy * px - vx * py)) / two_thirds, px / dist, py / dist]])


def computeCovMatrix(deltaT, sigma_aX, sigma_aY):
    A = np.array([[deltaT ** 2, 0],
                  [0, deltaT ** 2],
                  [deltaT, 0],
                  [0, deltaT]])
    P = np.array([[sigma_aX ** 2, 0],
                  [0, sigma_aY ** 2]])
    Q = (A.dot(P)).dot(A.T)

    return Q


def computeFmatrix(delta_t):
    F = np.array([[1, 0, delta_t, 0],
                  [0, 1, 0, delta_t],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    return F

def kalman_filter(measurements):
    xEstimate = []
    xTrue = []

    # Initial X_state.
    X_state_current = X_initial_state

    P = initialCovMatrix  # Initial P matrix
    I = np.eye(X_state_current.shape[0])  # Identity matrix
    acceleration = np.zeros((2, 1))
    last_dt = measurements[0][3] - 10000

    for i in range(1, len(measurements)):
        currentMeas = measurements[i]

        # Compute the current deltaT
        if (currentMeas[0] == 'L'):
            timeStamp = currentMeas[3]
            deltaT = timeStamp - last_dt
            last_dt = timeStamp
            # Ground-truth value
            X_true_current = np.array([[currentMeas[4]],  # PosX
                                       [currentMeas[5]],  # PoxY
                                       [currentMeas[6]],  # Vx
                                       [currentMeas[7]]])  # Vy

            # perform predict
            F_matrix = computeFmatrix(deltaT)
            X_state_current = F_matrix.dot(X_state_current)

            sig_ax, sig_ay = acceleration.flatten()  # Extract
            Q = computeCovMatrix(deltaT, sig_ax, sig_ay)

            # In order to zero all except for main diagonal
            #P = np.multiply((F_matrix.dot(P)).dot(F_matrix.T) + Q, I)
            P = (F_matrix.dot(P)).dot(F_matrix.T) + Q

            # perform measurement update
            last_v = X_state_current[2:]  # Save the predicted velocity
            z = np.array([[currentMeas[1]],  # Measured PosX
                          [currentMeas[2]]])  # Measured posY

            y = z - H_Lidar.dot(X_state_current)
            S = (H_Lidar.dot(P)).dot(H_Lidar.T) + R_lidar
            KG = (P.dot(H_Lidar.T)).dot(np.linalg.inv(S))  # Kalman Gain
            X_state_current = X_state_current + KG.dot(y)  # Updated state estimation
            P = (I - KG.dot(H_Lidar)).dot(P)  # Updated error state estimation
            # Update acceleration
            acceleration = X_state_current[2:] - last_v

            # Update plot data for visualization
            updatePlot(i, X_true_current, z, X_state_current, True)

        if (currentMeas[0] == 'L'):
            xEstimate.append(X_state_current)
            xTrue.append(X_true_current)
    return xEstimate, xTrue