import random
import numpy as np


def prime_q(n):
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


def generate_rand_matrix_1(n):
    return [[random.random() for i in range(n)] for j in range(n)]


def generate_rand_matrix_2(n):
    return np.random.uniform(size=(n, n))


def matrix_mul_py(a, b):
    n = len(a)
    c: list = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[i][j] = a[i][k] * b[k][j]
    return c


def matrix_mul_numpy(a, b):
    n = a.shape[0]
    if n != a.shape[1] or n != b.shape[0] or n != b.shape[1]:
        return None
    return np.matmul(a, b)


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


def jordan_method_py(a, f) -> None:
    n = len(a)
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


def jordan_method_numpy(A: np.array, r: np.array) -> np.array:
    size: int = A.shape[0]
    for i in range(0, size):
        r[i] = r[i] / A[i, i]
        A[i] = A[i] / A[i, i]
        for j in range(i + 1, size):
            r[j] = r[j] - r[i] * A[j, i]
            A[j] = A[j] - A[i] * A[j, i]
        for k in range(i - 1, -1, -1):
            r[k] = r[k] - r[i] * A[k, i]
            A[k] = A[k] - A[i] * A[k, i]
    return r
