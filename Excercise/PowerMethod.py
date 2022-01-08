import math

import numpy as np
from typing import List
"""
Execute power method
"""


def norm2(x):
    fac = math.sqrt(sum([x_i ** 2 for x_i in x]))
    x_n = x / fac
    return fac, x_n


def normalize(x):
    fac = abs(x).max()
    x_n = x / x.max()
    return fac, x_n


def power_method(initial_vector: List, matrix: List[List[float]], noi: int = 300) -> List:
    i = 0
    vector = np.array(initial_vector)  # create a vector array
    pagerank_matrix = np.array(matrix)
    eigenvalues = []
    eigenvectors = []# transition matrix
    # while max iteration don't reached or  consecutive vectors are close enough
    for i in range(noi):
        vector = np.dot(pagerank_matrix, vector)
        lambda_1, vector = normalize(vector)
        eigenvalues.append(lambda_1)
        eigenvectors.append(vector)
    print('Eigenvalues', lambda_1)
    print('Eigenvectors', vector)
    return 0


def pagerank(M, num_iterations: int = 100, d: float = 0.85):
    N = M.shape[1]
    v = np.random.rand(N, 1)
    v = v / np.linalg.norm(v, 1)
    M_hat = (d * M + (1 - d) / N)
    for i in range(num_iterations):
        v = M_hat @ v
    print(v)
    return v


M = np.array([[0, 0, 0, 0, 1],
              [0.5, 0, 0, 0, 0],
              [0.5, 0, 0, 0, 0],
              [0, 1, 0.5, 0, 0],
              [0, 0, 0.5, 1, 0]])
v = pagerank(M, 100, 0.85)

power_method([1, 70], [[0, 2], [2, 3]], 10)
