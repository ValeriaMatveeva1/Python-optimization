import math
import random
import numpy as np
from numba import njit
from numba.typed import List


@njit
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
def find_primes(n):
    a = []
    for i in range(n + 1):
        if prime_q(i):
            a.append(i)
    return a


@njit
def generate_rand_matrix(n):
    a = List([random.random() for i in range(n * n)])
    return a


@njit
def matrix_mul(a, b):
    n = int(math.sqrt(len(a)))
    c = List([0.0 for i in range(n * n)])
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[i * n + j] = a[i * n + k] * b[k * n + j]
    return c


@njit
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
def jordan_method_py(a: np.array, f: np.array) -> None:
    n = a.shape[0]
    for i in range(0, n):
        k = i
        for j in range(i + 1, n):
            if abs(a[j][i]) > abs(a[k][i]):
                k = j
        if abs(a[k][i]) < 10E-9:
            print("Non-invertible matrix")
            return
        for j in range(n):
            a[k][j], a[i][j] = a[i][j], a[k][j]
            f[k][j], f[i][j] = f[i][j], f[k][j]
        for j in range(n):
            f[i][j] /= a[i][i]
        for j in range(n - 1, -1, -1):
            a[i][j] /= a[i][i]

        for j in range(i + 1, n):
            for t in range(n):
                f[j][t] -= f[i][t] * a[j][i]
            for t in range(n - 1, i - 1, -1):
                a[j][t] -= a[i][t] * a[j][i]
        for j in range(i - 1, -1, -1):
            for t in range(n):
                f[j][t] -= f[i][t] * a[j][i]
            for t in range(n - 1, i - 1, -1):
                a[j][t] -= a[i][t] * a[j][i]
    return f


@njit
def jordan_method_numpy(a: np.array, r: np.array):
    n = a.shape[0]
    for i in range(0, n):
        r[i] = r[i] / a[i, i]
        a[i] = a[i] / a[i, i]
        for j in range(i + 1, n):
            r[j] = r[j] - r[i] * a[j, i]
            a[j] = a[j] - a[i] * a[j, i]
        for k in range(i - 1, -1, -1):
            r[k] = r[k] - r[i] * a[k, i]
            a[k] = a[k] - a[i] * a[k, i]
    return r
