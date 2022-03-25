import math
import random
import numpy as np
from numba import njit
from numba.typed import List


def prime_q(n):
    if n <= 1:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


@njit
def prime_q_1(n):
    if n <= 1:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def find_primes(n):
    a = []
    for i in range(n + 1):
        if prime_q(i):
            a.append(i)
    return a


@njit
def find_primes_1(n):
    a = []
    for i in range(n + 1):
        if prime_q_1(i):
            a.append(i)
    return a


def generate_rand_matrix_1(n):
    return [[random.random() for i in range(n)] for j in range(n)]


def generate_rand_matrix_2(n):
    return np.random.uniform(size=(n, n))


@njit
def generate_rand_matrix_3(n):
    a = List([random.random() for i in range(n * n)])

    return a


def matrix_mul_1(a, b):
    n = len(a)
    c: list = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[i][j] = a[i][k] * b[k][j]
    return c


def matrix_mul_2(a, b):
    n = a.shape[0]
    if n != a.shape[1] or n != b.shape[0] or n != b.shape[1]:
        return None
    return np.matmul(a, b)


@njit
def matrix_mul_3(a, b):
    n = int(math.sqrt(len(a)))
    c = List([0.0 for i in range(n * n)])
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[i * n + j] = a[i * n + k] * b[k * n + j]
    return c




def mandelbrot_set(wh):
    pmin, pmax, qmin, qmax = -2.5, 1.5, -2, 2
    ppoints, qpoints = wh, wh
    max_iterations = 100
    infinity_border = 100
    image = np.zeros((ppoints, qpoints))
    for ip, p in enumerate(np.linspace(pmin, pmax, ppoints)):
        for iq, q in enumerate(np.linspace(qmin, qmax, qpoints)):
            zx, zy = p, q
            for k in range(max_iterations):
                zx, zy = zx ** 2 - zy ** 2 + p, 2 * zx * zy + q
                if zx ** 2 + zy ** 2 > infinity_border:
                    image[ip, iq] = k
                    break
    return image


@njit
def mandelbrot_set_2(wh):
    pmin, pmax, qmin, qmax = -2.5, 1.5, -2, 2
    ppoints, qpoints = wh, wh
    max_iterations = 100
    infinity_border = 100
    image = np.zeros((ppoints, qpoints))
    for ip, p in enumerate(np.linspace(pmin, pmax, ppoints)):
        for iq, q in enumerate(np.linspace(qmin, qmax, qpoints)):
            zx, zy = p, q
            for k in range(max_iterations):
                zx, zy = zx ** 2 - zy ** 2 + p, 2 * zx * zy + q
                if zx ** 2 + zy ** 2 > infinity_border:
                    image[ip, iq] = k
                    break
    return image
