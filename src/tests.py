import random

import numpy as np

import algorithms as al
import algorithms_jit as alj
import algorithms_cython as alc
import time
import csv


# умножение матриц (обычные списки)
def test_1_1(n):
    a = al.generate_rand_matrix_1(n)
    b = al.generate_rand_matrix_1(n)
    st = time.time()
    al.matrix_mul_py(a, b)
    return time.time() - st


# умножение матриц (numpy-матрицы и умножение np.mul)
def test_1_2(n):
    a = al.generate_rand_matrix_2(n)
    b = al.generate_rand_matrix_2(n)
    st = time.time()
    al.matrix_mul_numpy(a, b)
    return time.time() - st


# умножение матриц (jit и обычные списки)
def test_1_3(n):
    a = alj.generate_rand_matrix(n)
    b = alj.generate_rand_matrix(n)
    st = time.time()
    alj.matrix_mul(a, b)
    return time.time() - st


# умножение матриц (cython и обычные списки)
def test_1_4(n):
    a = alc.generate_rand_matrix(n)
    b = alc.generate_rand_matrix(n)
    st = time.time()
    alc.matrix_mul(a, b)
    return time.time() - st


# множество Мандельброта (обычная реализация)
def test_2_1(n):
    st = time.time()
    al.mandelbrot_set(n)
    return time.time() - st


# множество Мандельброта (jit реализация)
def test_2_2(n):
    st = time.time()
    res = alj.mandelbrot_set(n)
    return time.time() - st


# множество Мандельброта (cython реализация)
def test_2_3(n):
    st = time.time()
    alc.mandelbrot_set(n)
    return time.time() - st


# нахождение всех простых чисел до n (обычная реализация)
def test_3_1(n):
    st = time.time()
    al.find_primes(n)
    return time.time() - st


# нахождение всех простых чисел до n (jit реализация)
def test_3_2(n):
    st = time.time()
    alj.find_primes(n)
    return time.time() - st


# нахождение всех простых чисел до n (cython реализация)
def test_3_3(n):
    st = time.time()
    alc.find_primes(n)
    return time.time() - st


# метод жордана (более медленная реализация)
def test_4_1(n):
    a = al.generate_rand_matrix_1(n)
    b = al.generate_rand_matrix_1(n)
    st = time.time()
    al.jordan_method_py(a, b)
    return time.time() - st


# метод жордана (хорошая реализация с numpy)
def test_4_2(n):
    a = al.generate_rand_matrix_2(n)
    b = al.generate_rand_matrix_2(n)
    st = time.time()
    al.jordan_method_numpy(a, b)
    return time.time() - st


# метод жордана (jit реализация с numpy)
def test_4_3(n):
    a = al.generate_rand_matrix_2(n)
    b = al.generate_rand_matrix_2(n)
    st = time.time()
    alj.jordan_method_py(a, b)
    return time.time() - st


# метод жордана (cython реализация)
def test_4_4(n):
    a = al.generate_rand_matrix_1(n)
    b = al.generate_rand_matrix_1(n)
    st = time.time()
    alc.jordan_method_py(a, b)
    return time.time() - st


# вычисление определителя (списка списков)
def test_5_1(n):
    a = al.generate_rand_matrix_1(n)
    st = time.time()
    al.determinant(a)
    return time.time() - st


# вычисление определителя (numpy)
def test_5_2(n):
    a = al.generate_rand_matrix_2(n)
    st = time.time()
    al.determinant_numpy(a)
    return time.time() - st


# вычисление определителя (jit)
def test_5_3(n):
    a = al.generate_rand_matrix_2(n)
    st = time.time()
    alj.determinant(a)
    return time.time() - st


# вычисление определителя (cpython)
def test_5_4(n):
    a = alc.generate_rand_matrix(n)
    st = time.time()
    alc.determinant(a)
    return time.time() - st


# вычисление определенного интеграла sin(x) (чистый python)
def test_6_1(n):
    st = time.time()
    al.integrate_py(0, n)
    return time.time() - st


# вычисление определенного интеграла sin(x) (numpy)
def test_6_2(n):
    st = time.time()
    al.integrate_numpy(0, n)
    return time.time() - st


# вычисление определенного интеграла sin(x) (jit)
def test_6_3(n):
    st = time.time()
    alj.integrate_py(0, n)
    return time.time() - st


# вычисление определенного интеграла sin(x) (cpython)
def test_6_4(n):
    st = time.time()
    alc.integrate_py(0, n)
    return time.time() - st


# интерполяция функции по точкам (чистый python)
def test_7_1(n):
    x = list(range(0, n, 1))
    y = [random.random() for _ in range(n)]
    x_new = [i / n for i in range(n)]
    st = time.time()
    al.interpolate_py(x, y, x_new)
    return time.time() - st


# интерполяция функции по точкам (numpy)
def test_7_2(n):
    x = np.arange(0, n, 1)
    y = np.random.uniform(size=(1, n))
    x_new = np.array([i / n for i in range(n)])
    st = time.time()
    al.interpolate_numpy(x, y, x_new)
    return time.time() - st


# интерполяция функции по точкам (jit)
def test_7_3(n):
    x = np.array([i for i in range(0, n, 1)])
    y = np.random.uniform(size=n)
    x_new = np.linspace(0, n, 100 * n)
    st = time.time()
    alj.interpolate_py(x, y, x_new)
    return time.time() - st


# интерполяция функции по точкам (cpython)
def test_7_4(n):
    x = list(range(0, n, 1))
    y = [random.random() for _ in range(n)]
    x_new = [i / n for i in range(n)]
    st = time.time()
    alc.interpolate_py(x, y, x_new)
    return time.time() - st


def write_tests(name: str, tests: list, b_l, b_r, step=1, rep=5):
    with open(name, mode='w') as w_file:
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")

        for i in range(b_l, b_r + 1, step):
            print(f"{round(100 * (i - b_l) / (b_r - b_l))}%")
            l = [i, ]
            for f in tests:
                t = 0
                for j in range(rep):
                    t += f(i)
                t /= rep
                l.append(t)
            file_writer.writerow(l)


def mean(name, new_name):
    with open(name, mode='r') as r_file:
        file_reader = csv.reader(r_file, delimiter=",", lineterminator="\r")
        with open(new_name, mode='w') as w_file:
            file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
            l1 = next(file_reader)
            l2 = next(file_reader)
            l3 = next(file_reader)
            i = next(file_reader)
            for i in file_reader:
                l = [l1[0], ]
                for j in range(1, len(l1)):
                    l.append((float(l1[j]) + float(i[j]) + float(l2[j]) + float(l3[j])) / 4)
                l1 = l2
                l2 = l3
                l3 = i
                file_writer.writerow(l)

# write_tests("data/determinant_calc.csv", [test_5_1, test_5_2, test_5_3, test_5_4], 1, 150)
# write_tests("data/integration.csv", [test_6_1, test_6_2, test_6_3, test_6_4], 0, 150)
# write_tests("data/interpolation.csv", [test_7_1, test_7_2, test_7_3, test_7_4], 2, 150)
# write_tests("data/matrix_mul.csv", [test_1_1, test_1_2, test_1_3, test_1_4], 1, 150)
# write_tests("data/mandelbrot_set.csv", [test_2_1, test_2_2, test_2_3], 1, 150)
# write_tests("data/find_primes.csv", [test_3_1, test_3_2, test_3_3], 10000, 11000)
# write_tests("data/jordan_method.csv", [test_4_1, test_4_2, test_4_3, test_4_4], 1, 100)
# mean("data/find_primes_o.csv", "data/find_primes.csv")
