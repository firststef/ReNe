import numpy as np


class MeanSquaredCost(object):  # Quadratic

    @staticmethod
    def value(y, t):
        return 0.5*len(y)*np.linalg.norm(y-t)**2  # aici era 0.5

    @staticmethod
    def error(y, t):
        fake_sigmoid_prime = y * (1 - y)  # normally you would use z to compute sigmoid(z)*(1 - sigmoid(z)) but y is sigmoid(z)
        return fake_sigmoid_prime * (y - t)


class CrossEntropyCost(object):

    @staticmethod
    def value(y, t):
        return np.sum(np.nan_to_num(-t*np.log(y)-(1-t)*np.log(1-y)))

    @staticmethod
    def error(y, t):
        return y - t
