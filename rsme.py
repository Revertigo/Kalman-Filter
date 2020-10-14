import numpy as np

def printRSME(EstimateVector, trueVector):
    rmse = computeRMSE(EstimateVector, trueVector)
    print("RSME:")
    print(rmse)

def computeRMSE(EstimateVector, trueVector):
    result = np.zeros((4, 1))
    for i in range(len(EstimateVector)):
        temp = np.square(np.subtract(EstimateVector[i], trueVector[i]))
        result = np.add(temp, result)

    result = result / len(EstimateVector)
    return np.sqrt(result)