import random
import numpy as np


def primeQ(n):
    if n <= 1:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def generate_rand_matrix_1(n):
    return [[random.random() for i in range(n)] for j in range(n)]


def matrix_mul_1(a, b):
    n = len(a)
    if len(a) != len(b) or len(a) != len(a[0]) or len(b) != len(b[0]):
        return None
    c = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[i][j] = a[i][k] * b[k][j]
    return c


def generate_rand_matrix_2(n):
    return np.random.uniform(size=(n, n))


def matrix_mul_2(a, b):
    n = a.shape[0]
    if n != a.shape[1] or n != b.shape[0] or n != b.shape[1]:
        return None
    c = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[i][j] = a[i][k] * b[k][j]
    return c
