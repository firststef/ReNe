import numpy as np
import unittest
import rnlib

matrix = [
    [1, 2, 3, 4],
    [11, 12, 13, 14],
    [21, 22, 23, 24]
]
array = [
    [2],
    [-5],
    [7],
    [-10]
]
base = [
    [0],
    [0],
    [0]
]


def numpy_1():
    print([x[-2:] for x in matrix[:2]])


def numpy_2():
    A = np.array(matrix)
    B = np.array([x[0] for x in array])
    print("A + B = ", A + B)
    print("A produs pe elemente B = ", A * B)
    print("A produs scalar B = ", A.dot(B))


def numpy_3():
    a = np.random.random((5, 5))
    print("a=", a.transpose())
    print("inv=", np.linalg.inv(a))
    print("det=", np.linalg.det(a))


if __name__ == "__main__":
    rnlib.get_matrix_mul(matrix, array)
    numpy_1()
    numpy_2()
    numpy_3()
    unittest.main()
