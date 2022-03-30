import random
import numpy as np
import math


def prime_q(n: int):
    if n <= 1:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def find_primes(n: int):
    a = []
    for i in range(n + 1):
        if prime_q(i):
            a.append(i)
    return a


def generate_rand_matrix_1(n: int):
    return [[random.random() for i in range(n)] for j in range(n)]


def generate_rand_matrix_2(n: int):
    return np.random.uniform(size=(n, n))


def matrix_mul_py(a: list, b: list):
    n = len(a)
    c: list = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[i][j] = a[i][k] * b[k][j]
    return c


def matrix_mul_numpy(a: np.array, b: np.array):
    n = a.shape[0]
    if n != a.shape[1] or n != b.shape[0] or n != b.shape[1]:
        return None
    return np.matmul(a, b)


def mandelbrot_set(wh: int):
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


def jordan_method_py(a: list, f: list):
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


def jordan_method_numpy(A: np.array, r: np.array):
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


def determinant(A):
    eps = 1e-9
    n = len(A)
    det = 1
    for i in range(n):
        k = i
        for j in range(i + 1, n):
            if abs(A[j][i]) > abs(A[k][i]):
                k = j
        if abs(A[k][i]) < eps:
            return 0
        A[i], A[k] = A[k], A[i]
        if i != k:
            det = -det
        det *= A[i][i]
        for j in range(i + 1, n):
            A[i][j] /= A[i][i]
        for j in range(n):
            if j != i and abs(A[j][i]) > eps:
                for t in range(i + 1, n):
                    A[j][t] -= A[i][t] * A[j][i]
    return det


def integrate(a: float, b: float):
    n = 500000
    h = (b - a) / float(n)
    total = sum(math.sin((a + (k * h))) for k in range(0, n))
    result = h * total
    return result


def interpolate_in_point(x: list, y: list, t: int):
    z = 0
    for j in range(len(y)):
        p1 = 1
        p2 = 1
        for i in range(len(x)):
            if i == j:
                p1 = p1 * 1
                p2 = p2 * 1
            else:
                p1 = p1 * (t - x[i])
                p2 = p2 * (x[j] - x[i])
        z = z + y[j] * p1 / p2
    return z


def interpolate(x: list, y: list, x_new: list):
    return [interpolate_in_point(x, y, i) for i in x_new]

print(determinant([[1,2,3],[0,-7,-3],[9,-8,2]]))