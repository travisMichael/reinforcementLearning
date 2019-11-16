import numpy as np


def snap_to_grid(val, minVal, maxVal, gridSize):
    # % This function will convert a given value to closest integer ranging
    # % from 1 to gridSize + 1.
    # % For example, minVal will be converted to 1 and maxVal will be
    # % converted to gridSize + 1.
    range = maxVal - minVal
    snappedVal = round((val - minVal) / range * gridSize)
    return np.min([int(snappedVal), gridSize])
