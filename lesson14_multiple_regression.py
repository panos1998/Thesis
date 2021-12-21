from lesson3 import dot, Vector


def predict(x: Vector, beta: Vector) -> float:
    return dot(x, beta)

from typing import List


def error(x: Vector, y: float, beta: Vector) -> float:
    return predict(x, beta) - y


def squared_error(x: Vector, y: float, beta: Vector) -> float:
    return error(x, y, beta) ** 2


x = [1, 2, 3]
y = 30
beta = [4, 4, 4]
assert error(x, y, beta) == -6
assert squared_error(x, y, beta) == 36


def sqerror_gradient(x: Vector, y: float, beta: Vector) -> Vector:
    err = error(x, y, beta)
    return [2 * err * x_i for x_i in x]


assert sqerror_gradient(x, y, beta) == [-12, -24, -36]
