import os
import time

import numpy as np
import matplotlib.pyplot as plt

# Add logger to support visualization
os.makedirs('log/', exist_ok=True)
logger = open('log/log_{}.txt'.format(np.floor(time.time()).astype(int)), 'w')
est_log = []
gt_log = []


# Calculation of single RSME value(uses for visualization)
def RMSE(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(a - b)).flatten()


def updateData(index, X_state_current, X_true_current):
    single_rsme = RMSE(X_state_current, X_true_current)
    # Add current state vector to the logger
    est_log.append(X_state_current)
    # Add ground-truth state vector to the logger
    gt_log.append(X_true_current)

    logger.write('{}:{}\n'.format(index, single_rsme))

    # In order to see live process
    if index % 5 == 0:
        # Save logs
        logger.flush()

    return single_rsme


def updatePlot(i, gt: np.ndarray, meas: np.ndarray, state: np.ndarray, disp_on=True) -> None:
    rmse = updateData(i, state, gt)
    # Plot
    plt.plot(gt[0], gt[1], 'g*')
    plt.plot(meas[0], meas[1], 'ro')
    plt.plot(state[0], state[1], 'b*')
    plt.legend(['Ground-Truth', 'Measurments', 'Kalman-Filter'], loc=2)
    plt.title("RMSE x={:.3},y={:.3},vx={:.3},vy={:.3}".format(*rmse))
    if disp_on:
        plt.pause(.01)


def finalizePlot():
    logger.close()
    error_log = np.array(est_log) - np.array(gt_log)
    f_rmse = np.sqrt(np.square(error_log).mean(0))
    plt.title("RMSE x={:.3},y={:.3},vx={:.3},vy={:.3}".format(*f_rmse.flatten()))
    plt.show()
