import numpy as np
from typing import List
"""
Execute power method
"""


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


power_method([1, 70], [[0, 2], [2, 3]], 10)