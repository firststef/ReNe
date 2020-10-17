import expression
import rnlib
import unittest
import re


def get_ec_matrix_from_file(name: str):
    matrix = [[] for y in range(3)]
    with open(name) as f:
        lines = f.readlines()
        for ln in range(3):
            line = lines[ln]

            x = expression.Variable("x")
            y = expression.Variable("y")
            z = expression.Variable("z")

            expr = eval(re.sub(r"([0-9]+)([a-z]|[A-Z])+", r"\1*\2", line).replace("=", "-(") + ")")

            matrix[ln] = [expr.get_var(x).factor, expr.get_var(y).factor, expr.get_var(z).factor]
    return matrix


class TestGetSolution(unittest.TestCase):
    def test_parse_line(self):
        x = expression.Variable("x")
        y = expression.Variable("y")
        z = expression.Variable("z")

        expr = eval("2*x - 3*y + 4*z - 4")
        self.assertEqual(expr.get_var("x").factor, 2)
        self.assertEqual(expr.get_var("y").factor, -3)
        self.assertEqual(expr.get_var("z").factor, 4)

    def test_parse_file(self):
        self.assertEqual(get_ec_matrix_from_file("file.txt"), [
            [1, 2, 1],
            [1, 0, 3],
            [2, -3, 0]
        ])


if __name__ == "__main__":
    unittest.main()