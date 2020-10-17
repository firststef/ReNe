import rnlib.functions as lf
import rnlib.expression as ex
import unittest
import numpy.testing as nt


class TestMatrices(unittest.TestCase):
    def test_matrix_multiply(self):
        matrix_1 = [[1, 2, 1], [0, 1, 0], [2, 3, 4]]
        matrix_2 = [[2, 5], [6, 7], [1, 8]]
        matrix_3 = [[15, 27], [6, 7], [26, 63]]

        self.assertEqual(matrix_3, lf.get_matrix_mul(matrix_1, matrix_2))

    def test_matrix_adjugate(self):
        matrix = [[-3, 2, -5], [-1, 0, -2], [3, -4, 1]]
        adjugate = lf.get_matrix_adjugate(matrix)

        self.assertEqual(adjugate, [
            [-8, 18, -4],
            [-5, 12, -1],
            [4, -6, 2]
        ])

        matrix = [[2, 3, 7], [-1, -1, 0], [0, 0, 12]]
        adjugate = lf.get_matrix_adjugate(matrix)
        self.assertEqual(adjugate, [
            [-12, -36,   7],
            [12,  24,  -7],
            [0,  -0,   1]
        ])

    def test_matrix_inverse(self):
        matrix = [[2, 3, 7], [-1, -1, 0], [0, 0, 12]]
        inverse = lf.get_matrix_inverse(matrix)
        try:
            nt.assert_array_almost_equal(inverse, [
                [-1, -3, 0.583333],
                [1, 2, -0.583333],
                [0, 0, 0.083333]
            ], decimal=3)
        except Exception as e:
            self.fail("Not the same")


class TestExpression(unittest.TestCase):
    def test_parse_line(self):
        x = ex.Variable("x")
        y = ex.Variable("y")
        z = ex.Variable("z")

        expr = eval("2*x - 3*y + 4*z - 4")
        self.assertEqual(expr.get_var("x").factor, 2)
        self.assertEqual(expr.get_var("y").factor, -3)
        self.assertEqual(expr.get_var("z").factor, 4)

    def test_parse_file(self):
        self.assertEqual(ex.get_ec_matrix_from_file("expr_test.txt"),
            (
                [[1, 2, 1],
                [1, 0, 3],
                [2, -3, 0]],
                [7,
                 11,
                 1]
             )
        )


if __name__ == "__main__":
    unittest.main()
