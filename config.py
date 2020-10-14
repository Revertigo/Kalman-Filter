import numpy as np

X_initial_state = np.array([[5.0],
                            [5.0],
                            [0],
                            [0]])
initialCovMatrix = np.diag([12, 12, 12, 12])  # Initial P matrix
H_Lidar = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]])  # Transformation matrix
R_lidar = np.diag([0.0225, 0.0225])