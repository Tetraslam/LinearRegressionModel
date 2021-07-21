import numpy as np


def computeCost(x, y, theta, m):
    a = 1 / (2 * m)
    b = np.sum(((x @ theta) - y) ** 2)
    j = a * b
    return j


def gradient(x, y, theta, m):
    alpha = 0.00001
    iteration = 2000

    j_history = np.zeros([iteration, 1])

    for i in range(0, 2000):
        error = (x @ theta) - y
        temp0 = theta[0] - ((alpha / m) * np.sum(error * x[:, 0]))
        temp1 = theta[1] - ((alpha / m) * np.sum(error * x[:, 1]))
        theta = np.array([temp0, temp1]).reshape(2, 1)
        j_history[i] = (1 / (2 * m)) * (np.sum(((x @ theta) - y) ** 2))

    return theta, j_history


