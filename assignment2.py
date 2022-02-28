"""
In this assignment you should find the intersection points for two functions.
"""
import math

import numpy as np
import time
import random
from collections.abc import Iterable

x_is_root = False
i = 0
MAX_ITER = 1000


def regula_falsi(f, range_start, range_end, maxerr):
    point = range_end
    f_end = f(range_end)
    f_point = f_end
    iter = 0
    while abs(f_point) >= maxerr and iter < MAX_ITER:
        point = range_end - f_end * ((range_end - range_start) / (f_end - f(range_start)))
        f_point = f(point)
        if (f_point < 0) != (f_end < 0):
            range_start = point
        else:
            range_end = point
            f_end = f_point
        iter += 1
    if abs(f_point) >= maxerr:
        return
    return point


def find_basic_roots(f, xs, n, maxerr):
    global i
    found_roots = []
    x0 = xs[i]
    f_x0 = f(x0)
    while abs(f_x0) < maxerr:
        found_roots.append(x0)
        i += 1
        if i >= n:
            return f_x0, x0, found_roots
        x0 = xs[i]
        f_x0 = f(x0)

    return f_x0, x0, found_roots


def intersect(f, a, b, maxerr):
    global i
    global x_is_root
    i = 0
    n = 100
    xs = np.linspace(a, b, n)

    f_x0, x0, found_roots = find_basic_roots(f, xs, n, maxerr)
    i += 1
    while i < n:
        x1 = xs[i]
        f_x1 = f(x1)
        if abs(f_x1) < maxerr:
            found_roots.append(x1)
            x_is_root = True
        elif (f_x0 > 0) == (f_x1 < 0):
            regula_falsi_root = regula_falsi(f, x0, x1, maxerr)
            if not regula_falsi_root is None:
                found_roots.append(regula_falsi_root)
                x_is_root = False
        x0 = x1
        f_x0 = f_x1
        i += 1
    return found_roots


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.

        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        # replace this line with your solution

        return intersect((lambda x: f1(x) - f2(x)), a, b, maxerr)

    ##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
        print("sqrt len = ", len(X))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(4)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
        print("poly len = ", len(X))

    def test_poly_unidentical(self):

        ass2 = Assignment2()

        f1 = lambda x: 4 * x * 4 + 3 * x * 3 - 2 * x ** 2 - x
        f2 = lambda x: x * 7 - x * 6 + x ** 5

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
        print("poly_unidentical len = ", len(X))

    def test_sinx_1_sinx(self):
        ass2 = Assignment2()

        f1 = np.sin
        f2 = lambda x: 1 / np.sin(x)

        X = ass2.intersections(f1, f2, -np.pi, np.pi, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
        print("sin_1/sin = ", len(X))

    def test_sinx_sin_1_x(self):
        ass2 = Assignment2()

        f1 = np.sin
        f2 = lambda x: np.sin(1 / x)

        X = ass2.intersections(f1, f2, 0.0001, 10, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
        print("sin_sin_1/x = ", len(X))

    def test_sin_cos(self):
        ass2 = Assignment2()

        f1 = np.sin
        f2 = np.cos

        X = ass2.intersections(f1, f2, -np.pi, np.pi, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
        print("sin_cos len = ", len(X))

    def test_sin_minus_sin(self):
        ass2 = Assignment2()

        f1 = np.sin
        f2 = lambda x: -np.sin(x)

        X = ass2.intersections(f1, f2, -np.pi, np.pi, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
        print("sin_minus_sin len = ", len(X))

    def test_x_sqrt_x(self):
        ass2 = Assignment2()

        f1 = lambda x: x - 1
        f2 = lambda x: np.sqrt(x - 1) if x >= 1 else 0

        X = ass2.intersections(f1, f2, 1, 25, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
        print("x_sqrt_x len = ", len(X))

    def test_sinx_x_3(self):
        ass2 = Assignment2()
        f1 = np.sin
        f2 = lambda x: x ** 3
        X = ass2.intersections(f1, f2, 1, 60, maxerr=0.001)
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
        print("sinx_x_3 len = ", len(X))


if __name__ == "_main_":
    unittest.main()
