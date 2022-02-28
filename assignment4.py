"""
In this assignment you should fit a model function of your choice to data
that you sample from a given function.

The sampled data is very noisy so you should minimize the mean least squares
between the model you fit and the data points you sample.

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You
must make sure that the fitting function returns at most 5 seconds after the
allowed running time elapses. If you take an iterative approach and know that
your iterations may take more than 1-2 seconds break out of any optimization
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools
for solving this assignment.

"""

import numpy as np
import time
import random
import itertools

epsilon = 0.00005


def make_coefficients(xs, d):
    length = len(xs)
    coefficients = np.identity(d + 1)
    coefficients[0][0] = length
    for i in range(d):
        coefficients[0][i + 1] = sum(xs[j] ** (i + 1) for j in range(length))
    for i in range(d):
        for j in range(d + 1):
            coefficients[i + 1][j] = sum(xs[k] ** (j + i + 1) for k in range(length))
    return coefficients


def make_solutions(xs, ys, d):
    solutions = []
    for i in range(d + 1):
        solutions.append(sum((xs[j] ** i) * ys[j] for j in range(len(xs))))
    return solutions


def gaussian_elimination_solver(a, b):
    n = len(b)
    x = np.zeros(n, float)

    for k in range(n - 1):
        if np.fabs(a[k, k]) < 1.0e-12:

            for i in range(k + 1, n):
                if np.fabs(a[i, k]) > np.fabs(a[k, k]):
                    a[[k, i]] = a[[i, k]]
                    b[[k, i]] = b[[i, k]]
                    break

        for i in range(k + 1, n):
            if a[i, k] == 0:
                continue

            factor = a[k, k] / a[i, k]
            for j in range(k, n):
                a[i, j] = a[k, j] - a[i, j] * factor
            b[i] = b[k] - b[i] * factor

    x[n - 1] = b[n - 1] / a[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        sum_ax = 0

        for j in range(i + 1, n):
            sum_ax += a[i, j] * x[j]

        x[i] = (b[i] - sum_ax) / a[i, i]

    return x


class Assignment4A:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        f : callable.
            A function which returns an approximate (noisy) ys_vector value given xs_vector.
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        a function:float->float that fits f between a and b
        """

        # replace these lines with your solution
        xs_vector = []
        ys_vector = []
        x = a
        measure_start = time.time()
        fx = f(x)
        measure_end = time.time()
        measure_time = (measure_end - measure_start) + epsilon
        # to avoid dividing by zero and to make sure I'll not pass the maxtime I added a small amount (epsilon)
        n = (maxtime * 0.6) / measure_time
        if n >= 1:
            points_diff = (b - a) / (n - 1)
            while x < b:
                xs_vector.append(x)
                ys_vector.append(f(x))
                x += points_diff

        if d > 12:
            d = 12
        coefficients = make_coefficients(xs_vector, d)
        solutions = make_solutions(xs_vector, ys_vector, d)
        try:
            solved_matrix = gaussian_elimination_solver(coefficients, solutions)
            solved_matrix = solved_matrix[::-1]
            solution = np.poly1d(solved_matrix)
        except:
            solution = np.poly1d([])

        return solution


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1, 1, 1))
        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(7)(NOISY(0.01)(poly(1, 1, 1)))

        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1, 1, 1)
        nf = NOISY(1)(f)
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse = 0
        for x in np.linspace(0, 1, 1000):
            self.assertNotEquals(f(x), nf(x))
            mse += (f(x) - ff(x)) ** 2
        mse = mse / 1000
        print(mse)


if __name__ == "__main__":
    unittest.main()
