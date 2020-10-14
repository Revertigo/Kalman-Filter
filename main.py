import pandas as pd
from algorithm import kalman_filter
from vizualization import *
from rsme import printRSME

if __name__ == '__main__':
    my_cols = ["A", "B", "C", "D", "E", "f", "g", "h", "i", "j", "k"]
    data = pd.read_csv("asset/data.txt", names=my_cols, delim_whitespace=True, header=None)

    measurements = []
    for index in range(0, len(data)):
        measurements.append(data.iloc[index, :].values)

    xEstimate, xTrue = kalman_filter(measurements)

    printRSME(xEstimate, xTrue)

    # Finalizing plot step
    finalizePlot()