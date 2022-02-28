"""
In this assignment you should interpolate the given function.
"""
import operator
from functools import reduce

import numpy as np
import time
import random
import sampleFunctions


def bisect_right(array, val):
    low = 0
    high = len(array)
    while low < high:
        mid = low + (high - low) // 2
        if array[mid] > val:
            high = mid
        else:
            low = mid + 1
    return low


def compute_right_side(n, diffs, ys):
    right_side = np.append([0], [6 * ((ys[i + 1] - ys[i]) / diffs[i] - (ys[i] - ys[i - 1]) / diffs[i - 1])
                                 / (diffs[i] + diffs[i - 1]) for i in range(1, n - 1)])
    right_side = np.append(right_side, [0])
    return right_side


def triagonal_matrix_solver(a, b, c, d):
    n = len(d)
    if n > 0:
        w = np.zeros(n - 1)
    else:
        w = np.zeros(0)
    g = np.zeros(n)
    p = np.zeros(n)

    w[0] = c[0] / b[0]
    g[0] = d[0] / b[0]

    for i in range(1, n - 1):
        w[i] = c[i] / (b[i] - a[i - 1] * w[i - 1])
    for i in range(1, n):
        g[i] = (d[i] - a[i - 1] * g[i - 1]) / (b[i] - a[i - 1] * w[i - 1])
    p[n - 1] = g[n - 1]
    for i in range(n - 1, 0, -1):
        p[i - 1] = g[i - 1] - w[i - 1] * p[i]
    return p


def compute_spline(xs_vector, ys_vector):
    n = len(xs_vector)
    diff_array = [xs_vector[i + 1] - xs_vector[i] for i in range(len(xs_vector) - 1)]

    bottom_diagonal = [diff_array[i] / (diff_array[i] + diff_array[i + 1]) for i in range(n - 2)] + [0]
    main_diagonal = [2] * n
    upper_diagonal = [0] + [diff_array[i + 1] / (diff_array[i] + diff_array[i + 1]) for i in range(n - 2)]
    right_side = compute_right_side(n, diff_array, ys_vector)

    solved_matrix = triagonal_matrix_solver(bottom_diagonal, main_diagonal, upper_diagonal, right_side)

    coefficients = [[(solved_matrix[i + 1] - solved_matrix[i]) * diff_array[i] * diff_array[i] / 6, solved_matrix[i] * diff_array[i] * diff_array[i] / 2,
                     (ys_vector[i + 1] - ys_vector[i] - (solved_matrix[i + 1] + 2 * solved_matrix[i]) * diff_array[i] * diff_array[i] / 6),
                     ys_vector[i]] for i in range(n - 1)]

    def spline_function(val):
        bisection = bisect_right(xs_vector, val) - 1
        idx = min(bisection, n - 2)
        z = (val - xs_vector[idx]) / diff_array[idx]
        C = coefficients[idx]
        return (((C[0] * z) + C[1]) * z + C[2]) * z + C[3]

    return spline_function


def spline_solver(f, a, b, n):
    xs_vector = np.linspace(a, b, n)
    ys_vector = [f(x) for x in xs_vector]
    interpolated_function = compute_spline(xs_vector, ys_vector)
    return interpolated_function


def lagrange_solver(f, a, b, n):
    xs_vector = []
    ys_vector = []
    points_diff = (b - a) / (n - 1)
    x = a
    for i in range(n):
        xs_vector.append(x)
        ys_vector.append(f(x))
        x += points_diff

    def lagrange_function(val):
        def lagrange(i):
            p = [(val - xs_vector[j]) / (xs_vector[i] - xs_vector[j]) for j in range(num_of_xs) if j != i]
            return reduce(operator.mul, p)

        num_of_xs = len(xs_vector)
        return sum(lagrange(j) * ys_vector[j] for j in range(num_of_xs))

    return lagrange_function


class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        starting to interpolate arbitrary functions.
        """

        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time.
        The assignment will be tested on variety of different functions with
        large n values.

        Interpolation error will be measured as the average absolute error at
        2*n random points between a and b. See test_with_poly() below.

        Note: It is forbidden to call f more than n times.

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.**

        Note: sometimes you can get very accurate solutions with only few points,
        significantly less than n.

        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """

        # replace this line with your solution to pass the second test

        if n <= 50:
            return lagrange_solver(f, a, b, n)
        return spline_solver(f, a, b, n)


##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):
    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        num_of_points = 100
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, num_of_points)

            xs = np.random.random(2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("x^30 polynomial: " + str(T) + "[s]")
        print("x^30 polynomial: " + str(mean_err) + "[mean_err]")

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)

    def test_with_sin(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 100
        for i in tqdm(range(100)):

            f = np.sin

            ff = ass1.interpolate(f, -10, 10, num_of_points)

            xs = np.random.uniform(low=-10, high=10, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("sin(x): " + str(T) + "[s]")
        print("sin(x): " + str(mean_err) + "[mean_err]")

    def test_with_y_5(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 100
        for i in tqdm(range(100)):

            f = lambda x: 5

            ff = ass1.interpolate(f, -10, 10, num_of_points)

            xs = np.random.uniform(low=-10, high=10, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("y=5: " + str(T) + "[s]")
        print("y=5: " + str(mean_err) + "[mean_err]")

    def test_with_sin_x_2(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 50
        for i in tqdm(range(100)):

            f = lambda x: np.sin(x ** 2)

            ff = ass1.interpolate(f, -1, 5, num_of_points)

            xs = np.random.uniform(low=-1, high=5, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("sin(x^2): " + str(T) + "[s]")
        print("sin(x^2): " + str(mean_err) + "[mean_err]")

    def test_with_e_with_exponent(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 100
        for i in tqdm(range(100)):

            f = lambda x: np.exp(-2 * (x ** 2))

            ff = ass1.interpolate(f, -2, 4, num_of_points)

            xs = np.random.uniform(low=-2, high=4, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("e^(-2x^2): " + str(T) + "[s]")
        print("e^(-2x^2): " + str(mean_err) + "[mean_err]")

    def test_with_arctan(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 100
        for i in tqdm(range(100)):

            f = lambda x: np.arctan(x)

            ff = ass1.interpolate(f, -5, 5, num_of_points)

            xs = np.random.uniform(low=-5, high=5, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("arctan: " + str(T) + "[s]")
        print("arctan: " + str(mean_err) + "[mean_err]")

    def test_with_sinx_div_x(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 100
        for i in tqdm(range(100)):

            f = lambda x: np.sin(x) / x

            ff = ass1.interpolate(f, 0.00001, 10, num_of_points)

            xs = np.random.uniform(low=0.00001, high=10, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("sin(x)/x: " + str(T) + "[s]")
        print("sin(x)/x: " + str(mean_err) + "[mean_err]")

    def test_with_1_div_lnx(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 200
        for i in tqdm(range(100)):

            f = lambda x: 1 / np.log(x)

            ff = ass1.interpolate(f, 0.00001, 0.9999, num_of_points)

            xs = np.random.uniform(low=0.00001, high=1, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("1/ln(x): " + str(T) + "[s]")
        print("1/ln(x): " + str(mean_err) + "[mean_err]")

    def test_with_e_e_x(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 200
        for i in tqdm(range(100)):

            f = lambda x: np.exp(np.exp(x))

            ff = ass1.interpolate(f, -2, 2, num_of_points)

            xs = np.random.uniform(low=-2, high=2, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("e^e^x: " + str(T) + "[s]")
        print("e^e^x: " + str(mean_err) + "[mean_err]")

    def test_with_ln_ln_x(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 100
        for i in tqdm(range(100)):

            f = lambda x: np.log(np.log(x))

            ff = ass1.interpolate(f, 1.00001, 30, num_of_points)

            xs = np.random.uniform(low=1.00001, high=30, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("ln(ln(x)): " + str(T) + "[s]")
        print("ln(ln(x)): " + str(mean_err) + "[mean_err]")

    def test_with_a_polynomial(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 100
        for i in tqdm(range(100)):

            f = lambda x: 5 * (x ** 2) - 10 * x + 1

            ff = ass1.interpolate(f, 3, 10, num_of_points)

            xs = np.random.uniform(low=3, high=10, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("5x^2-10x+1:" + str(T) + "[s]")
        print("5x^2-10x+1: " + str(mean_err) + "[mean_err]")

    def test_with_a_exp_sin(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 100
        for i in tqdm(range(100)):

            f = lambda x: 2 * (1 / (x * 2)) * np.sin(1 / x)

            ff = ass1.interpolate(f, 3, 10, num_of_points)

            xs = np.random.uniform(low=3, high=10, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("2^(1/x^2 )*sin(1/x):" + str(T) + "[s]")
        print("2^(1/x^2 )*sin(1/x): " + str(mean_err) + "[mean_err]")

    def test_with_sin_ln_x(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 100
        for i in tqdm(range(100)):

            f = lambda x: np.sin(np.log(x))

            ff = ass1.interpolate(f, 3, 10, num_of_points)

            xs = np.random.uniform(low=3, high=10, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("sin(ln(x):" + str(T) + "[s]")
        print("sin(ln(x): " + str(mean_err) + "[mean_err]")

    def test_with_e_ln(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 100
        for i in tqdm(range(100)):

            f = lambda x: np.log(np.exp(x))

            ff = ass1.interpolate(f, -10, 10, num_of_points)

            xs = np.random.uniform(low=-10, high=10, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("ln(e^x):" + str(T) + "[s]")
        print("ln(e^x): " + str(mean_err) + "[mean_err]")

    def test_with_ln_e(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 100
        for i in tqdm(range(100)):

            f = lambda x: np.exp(np.log(x))

            ff = ass1.interpolate(f, 0.0000001, 10, num_of_points)

            xs = np.random.uniform(low=0.0000001, high=10, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("e^(lnx):" + str(T) + "[s]")
        print("e^(lnx): " + str(mean_err) + "[mean_err]")

    def test_with_poly_div_sin(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 100
        for i in tqdm(range(100)):

            f = lambda x: (pow(2, (1 / (x ** 2)))) * (np.sin(1 / x))

            ff = ass1.interpolate(f, 3, 10, num_of_points)

            xs = np.random.uniform(low=3, high=10, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("2^(1/(x^2))*sin(1/x): " + str(T) + "[s]")
        print("2^(1/(x^2))*sin(1/x):  " + str(mean_err) + "[mean_err]")


if __name__ == "_main_":
    unittest.main()
