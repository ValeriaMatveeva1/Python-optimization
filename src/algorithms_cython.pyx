import math
import random


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


def generate_rand_matrix(n):
    return [[random.random() for i in range(n)] for j in range(n)]


def matrix_mul(a, b):
    n = len(a)
    c: list = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[i][j] = a[i][k] * b[k][j]
    return c


def linspace(min, max, step):
    h = (max-min)/step
    while step:
        yield min
        step-=1
        min += h


def mandelbrot_set(wh):
    pmin, pmax, qmin, qmax = -2.5, 1.5, -2, 2
    ppoints, qpoints = wh, wh
    max_iterations = 100
    infinity_border = 100
    image = [[0]*qpoints for i in range(ppoints)]
    for ip, p in enumerate(linspace(pmin, pmax, ppoints)):
        for iq, q in enumerate(linspace(qmin, qmax, qpoints)):
            zx, zy = p, q
            for k in range(max_iterations):
                zx, zy = zx ** 2 - zy ** 2 + p, 2 * zx * zy + q
                if zx ** 2 + zy ** 2 > infinity_border:
                    image[ip][iq] = k
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
