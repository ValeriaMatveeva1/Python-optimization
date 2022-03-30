import math
import random


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


def generate_rand_matrix(n: int):
    return [[random.random() for i in range(n)] for j in range(n)]


def matrix_mul(a: list, b: list):
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


def mandelbrot_set(wh: int):
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


def jordan_method_py(a: list, f: list):
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