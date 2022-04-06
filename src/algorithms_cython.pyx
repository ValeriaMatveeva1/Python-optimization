import math
import random
import numpy as np
from libc.math cimport sin

cdef int prime_q(int n):
    if n <= 1:
        return 0
    cdef i = 2
    while i * i <= n:
        if n % i == 0:
            return 0
        i += 1
    return 1


cpdef int[:] find_primes(int n):
    cdef int[:] a = np.zeros(n+1, dtype=np.int32)
    cdef int i
    for i in range(n + 1):
        a[i] = prime_q(i)
    return a


cpdef double[:,:] generate_rand_matrix(int n):
    cdef double[:,:] a = np.random.uniform(size=(n, n))
    return a


cpdef double[:,:] matrix_mul(double[:,:] a, double[:,:] b):
    cdef int n = a.shape[0]
    cdef double[:,:] c = np.zeros((n, n))
    cdef int i, j, k
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[i,j] = a[i,k] * b[k,j]
    return c

cpdef int[:,:] mandelbrot_set(int wh):
    cdef double pmin = -2.5
    cdef double pmax = 1.5
    cdef double qmin = -2
    cdef double qmax = 2
    cdef int max_iterations = 100
    cdef int infinity_border = 100
    cdef int[:,:] image = np.zeros((wh, wh), dtype=np.int32)
    cdef int ip, iq
    cdef double p = pmin, q = qmin, zx, zy,
    cdef double hx = (pmax-pmin)/(wh*1.0), hy = (qmax-qmin)/(wh*1.0)
    for ip in range(wh):
        q = qmin
        for iq in range(wh):
            zx = p
            zy = q
            for k in range(max_iterations):
                zx, zy = zx ** 2 - zy ** 2 + p, 2 * zx * zy + q
                if zx ** 2 + zy ** 2 > infinity_border:
                    image[ip,iq] = k
                    break
            q += hy
        p += hx
    return image


cpdef double[:,:] jordan_method_py(double[:,:] a, double[:,:] f):
    cdef int n = a.shape[0]
    cdef int i, j, k
    for i in range(n):
        k = i
        for j in range(i + 1, n):
            if abs(a[j,i]) > abs(a[k,i]):
                k = j
        if abs(a[k,i]) < 10E-9:
            return f
        for j in range(n):
            a[k,j], a[i,j] = a[i,j], a[k,j]
            f[k,j], f[i,j] = f[i,j], f[k,j]
        for j in range(n):
            f[i,j] /= a[i,i]
        for j in range(n - 1, -1, -1):
            a[i,j] /= a[i,i]

        for j in range(i + 1, n):
            for t in range(n):
                f[j,t] -= f[i,t] * a[j,i]
            for t in range(n - 1, i - 1, -1):
                a[j,t] -= a[i,t] * a[j,i]
        for j in range(i - 1, -1, -1):
            for t in range(n):
                f[j,t] -= f[i,t] * a[j,i]
            for t in range(n - 1, i - 1, -1):
                a[j,t] -= a[i,t] * a[j,i]
    return f

cpdef double determinant(double[:,:] A) except? 0:
    cdef double eps = 1e-9
    cdef int n = A.shape[0]
    cdef double det = 1
    cdef int i, j, k
    for i in range(n):
        k = i
        for j in range(i + 1, n):
            if abs(A[j,i]) > abs(A[k,i]):
                k = j
        if abs(A[k,i]) < eps:
            return 0
        A[i], A[k] = A[k], A[i]
        if i != k:
            det = -det
        det *= A[i,i]
        for j in range(i + 1, n):
            A[i,j] /= A[i,i]
        for j in range(n):
            if j != i and abs(A[j,i]) > eps:
                for t in range(i + 1, n):
                    A[j,t] -= A[i,t] * A[j,i]
    return det


cpdef double integrate_py(double a, double b):
    cdef int n = 500000
    cdef double h0 = (b - a) / (n*1.0)
    cdef double h = h0
    cdef double total = 0
    cdef int i
    for i in range(n):
        total += sin(a + h)
        h += h0
    result = h0 * total
    return result


cdef double interpolate_in_point(double[:] x, double[:] y, double t):
    cdef double z = 0
    cdef int i, j
    cdef double p1
    cdef double p2
    for j in range(len(y)):
        p1 = 1.0
        p2 = 1.0
        for i in range(len(x)):
            if i == j:
                p1 = p1 * 1
                p2 = p2 * 1
            else:
                p1 = p1 * (t - x[i])
                p2 = p2 * (x[j] - x[i])
        z = z + y[j] * p1 / p2
    return z


cpdef double[:] interpolate_py(double[:] x, double[:] y, double[:] x_new):
    cdef int n = x_new.shape[0]
    cdef double[:] res = np.zeros(n)
    cdef int i
    for i in range(n):
        res[i] = interpolate_in_point(x, y, x_new[i])
    return res