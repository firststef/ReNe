import rnlib.expression as ex
import rnlib.functions as fn
import re
import unittest
import numpy as np
import numpy.testing as nt


class CompareOwnAndNumpy(unittest.TestCase):
    def test_compare(self):
        mat, res = ex.get_ec_matrix_from_file("expr.txt")
        self.assertEqual(
            [[12, 1, -7],
             [9, 0, 3],
             [5, 4, 8]],
            mat
        )
        solution_py = fn.get_matrix_mul(fn.get_matrix_inverse(mat), res)

        solution_numpy_instant = np.linalg.solve(mat, res)
        solution_numpy_manual = np.dot(np.linalg.inv(mat), res)

        try:
            nt.assert_array_almost_equal(solution_py, solution_numpy_instant)
            nt.assert_array_almost_equal(solution_py, solution_numpy_manual)
        except Exception:
            self.fail("The two matrices are not almost equal")


if __name__ == "__main__":
    unittest.main()
