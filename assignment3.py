"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and
the leftmost intersection points of the two functions.

The functions for the numeric answers are specified in MOODLE.


This assignment is more complicated than Assignment1 and Assignment2 because:
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors.
    2. You have the freedom to choose how to calculate the area between the two functions.
    3. The functions may intersect multiple times. Here is an example:
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately.
       You should explain why in one of the theoretical questions in MOODLE.

"""

import numpy as np
import time
import random
import assignment2


def simpson13(f, a, b, n):
    h = (b - a) / n
    xs = np.linspace(a, b, n + 1)
    integration = f(a) + f(b)
    for i in range(1, n):
        if i % 2 == 0:
            integration = integration + 2 * np.float32(f(xs[i]))
        else:
            integration = integration + 4 * np.float32(f(xs[i]))
    integration = integration * h / 3

    return np.float32(integration)


def midpoint_rule(f, a, b, n):
    dx = (b - a) / n
    x = np.linspace(a, b, n + 1)
    x_mid = (x[:-1] + x[1:]) / 2
    array = [f(x_mid[i]) * dx for i in range(len(x_mid))]
    return np.float32(np.sum(array))


class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n
        points. Your main objective is minimizing the integration error.
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions.

        Integration error will be measured compared to the actual value of the
        definite integral.

        Note: It is forbidden to call f more than n times.

        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """

        if a > b:
            if n % 2 == 0:
                if n == 2:
                    return -midpoint_rule(f, b, a, n)
                return -simpson13(f, b, a, n - 2)
            else:
                if n == 1:
                    return -midpoint_rule(f, b, a, n)
                return -simpson13(f, b, a, n - 1)
        else:
            if n % 2 == 0:
                if n == 2:
                    return midpoint_rule(f, a, b, n)
                return simpson13(f, a, b, n - 2)
            else:
                if n == 1:
                    return midpoint_rule(f, a, b, n)
                return simpson13(f, a, b, n - 1)

    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the area enclosed between two functions. This method finds
        all intersection points between the two functions to work correctly.

        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area.

        In order to find the enclosed area the given functions must intersect
        in at least two points. If the functions do not intersect or intersect
        in less than two points this function returns NaN.
        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """

        # replace this line with your solution
        as2 = assignment2
        new_f = (lambda x: f1(x) - f2(x))
        points_of_intersections = as2.intersect(new_f, 1.0, 100.0, maxerr=0.001)
        if not isinstance(points_of_intersections, list):
            points_of_intersections = [points_of_intersections]
        n = len(points_of_intersections)
        if n < 2:
            return np.float32(np.nan)

        area = sum(np.abs(self.integrate(new_f, points_of_intersections[i], points_of_intersections[i + 1], 501))
                   for i in range(n - 1))
        return np.float32(area)


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm
import math


class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        r = ass3.integrate(f1, -1, 1, 101)
        self.assertEqual(r.dtype, np.float32)

    def test_integral_x(self):
        ass3 = Assignment3()
        f1 = lambda x: x
        r = ass3.integrate(f1, 0, 1, 101)
        true_result = 0.5
        print("result of integrate x = ", r, "\t", "expect: ", true_result)
        # self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_integral_sqrt_x(self):
        ass3 = Assignment3()
        f = lambda x: np.sqrt(x) if x >= 0 else 0
        r = ass3.integrate(f, 0, 1, 101)
        true_result = 0.66667
        print("result of integrate sqrt x = ", r, "\t", "expect: ", true_result)
        # self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_area_x_with_sqrt_x(self):
        ass3 = Assignment3()
        f1 = lambda x: x - 1
        f2 = lambda x: np.sqrt(x - 1) if x >= 1 else 0
        r = ass3.areabetween(f1, f2)
        true_result = 0.166667
        print("result of erea sqrt x with x = ", r, "\t", "expect: ", true_result)
        # self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_integral_sin_x(self):
        ass3 = Assignment3()
        f1 = np.sin
        r = ass3.integrate(f1, -np.pi, np.pi, 101)
        print("result of integrate sinx = ", r, "\t", "expect: ", 0.0)
        # self.assertGreaterEqual(0.00000001, np.abs(r))

    def test_area_sinx_x3(self):
        ass3 = Assignment3()
        f1 = np.sin
        f2 = lambda x: x ** 3
        r = ass3.areabetween(f1, f2)
        print("result of area sin x with x^3  = ", r, "\t", "expect: ", np.NaN)
        self.assertTrue(math.isnan(r))

    def test_area_ex_poly(self):
        ass3 = Assignment3()
        f1 = lambda x: np.e ** x
        f2 = lambda x: x ** 4 - x ** 3 + x ** 2
        r = ass3.areabetween(f1, f2)
        true_result = 2875.67
        print("result of area e^x x with x^4-x^3+x^2 = ", r, "\t", "expect: ", true_result)
        # self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_integral_e_sinx(self):
        ass3 = Assignment3()
        f1 = lambda x: np.e ** np.sin(x)
        r = ass3.integrate(f1, -10, 10, 101)
        true_result = 25.0755
        print("result of integrate e^sinx = ", r, "\t", "expect: ", true_result)
        # self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_integral_8_x(self):
        ass3 = Assignment3()
        f1 = exp(8)
        r = ass3.integrate(f1, 0, 1.5, 101)
        true_result = 10.4006
        print("result of integrate 8^x = ", r, "\t", "expect: ", true_result)
        # self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.09, 10, 28500)
        true_result = -7.78662 * 10 ** 33
        print("result of hard case = ", r, "\t", "expect: ", true_result)
        # self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_areabetween(self):
        ass3 = Assignment3()
        # f1 = np.poly1d([2, 0, 0, 0, 0])
        # f2 = np.poly1d([1, 9, 0, 0, 0])
        f1 = np.sin
        f2 = np.cos
        r = ass3.areabetween(f1, f2)
        true_result = 84.85
        # self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))


if __name__ == "__main__":
    unittest.main()
