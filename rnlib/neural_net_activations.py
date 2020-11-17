import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    return np.exp(z) / sum(np.exp(z))
