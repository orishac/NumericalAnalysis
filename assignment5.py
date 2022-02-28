"""
In this assignment you should fit a model function of your choice to data
that you sample from a contour of given shape. Then you should calculate
the area of that shape.

The sampled data is very noisy so you should minimize the mean least squares
between the model you fit and the data points you sample.

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You
must make sure that the fitting function returns at most 5 seconds after the
allowed running time elapses. If you know that your iterations may take more
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment.
Note: !!!Despite previous note, using reflection to check for the parameters
of the sampled function is considered cheating!!! You are only allowed to
get (x,y) points from the given shape by calling sample().
"""
import array

import numpy as np
import time
import random
from scipy.signal import savgol_filter
from functionUtils import AbstractShape

epsilon = 0.00005


def trapezoidal(a, b):
    return (b[0] - a[0]) * (b[1] + a[1])


def area_calculator(points):
    n = len(points)
    area = [trapezoidal(points[i], points[i + 1]) for i in range(n - 1)]
    area.append(trapezoidal(points[n - 1], points[0]))
    return 0.5 * abs(sum(area))


class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self, samples):
        self.samples = samples
        pass

    def area(self):
        area = area_calculator(self.samples)
        return area


class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def area(self, contour: callable, maxerr=0.001) -> np.float32:
        """
        Compute the area of the shape with the given contour.

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """

        i = 5
        n = 2000
        shape = contour(n)
        new_maxerr = maxerr * 2
        first_area = area_calculator(shape[0:len(shape):i + 1])
        while new_maxerr > 0.5 * maxerr and i > 0:
            second_area = area_calculator(shape[0:len(shape):i])
            relative_error = abs(second_area - first_area) / second_area
            new_maxerr = relative_error
            i = i - 1
            first_area = second_area
        if i == 0:
            shape = contour(10000)
            second_area = area_calculator(shape)
        return second_area

    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        sample : callable.
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        An object extending AbstractShape.
        """

        # replace these lines with your solution
        start = time.time()
        sample()
        end = time.time()
        measure_time = end - start + epsilon
        n = (maxtime * 0.4 / measure_time)
        n = int(n)
        samples = []
        for i in range(n):
            samples.append(sample())

        if len(samples) == 0:
            return MyShape(None)

        samples = np.array(samples)
        new_samples = (samples - samples.mean(0))
        samples_sorted = new_samples[np.angle((new_samples[:, 0] + 1j * new_samples[:, 1])).argsort()]

        xs_by_angle = [x[0] for x in samples_sorted]
        ys_by_angle = [x[1] for x in samples_sorted]
        new_xs = savgol_filter(xs_by_angle, 51, 10)
        new_ys = savgol_filter(ys_by_angle, 51, 10)
        xy = np.stack((new_xs, new_ys), axis=1)
        my_shape = MyShape(xy)

        return my_shape


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(7)
            return circ()

        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        print("the area is: " + str(a))
        print("the expected area is: " + str(np.pi))
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_circle_bigger_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=2, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        print("the area is: " + str(a))
        print("the expected area is: " + str(4 * np.pi))
        self.assertLess(abs(a - 4 * np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_circle_even_bigger_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=10, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        print("the area is: " + str(a))
        print("the expected area is: " + str(100 * np.pi))
        self.assertLess(abs(a - 100 * np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        print("the area is: " + str(a))
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_circle_area_from_contour(self):
        circ = Circle(cx=1, cy=1, radius=1, noise=0.0)
        ass5 = Assignment5()
        T = time.time()
        a_computed = ass5.area(contour=circ.contour, maxerr=0.1)
        T = time.time() - T
        a_true = circ.area()
        print("The expected area is: " + str(a_true))
        print("the area found is: " + str(a_computed))
        self.assertLess(abs((a_true - a_computed) / a_true), 0.1)

    def test_circle_area_from_contour_2(self):
        circ = Circle(cx=2, cy=1, radius=2, noise=0.0)
        ass5 = Assignment5()
        T = time.time()
        a_computed = ass5.area(contour=circ.contour, maxerr=0.1)
        T = time.time() - T
        a_true = circ.area()
        print("The expected area is: " + str(a_true))
        print("the area found is: " + str(a_computed))
        self.assertLess(abs((a_true - a_computed) / a_true), 0.1)

    # def test_polygon_area(self):
    #     polygon = noisy_polygon(
    #         knots=[(0, 0), (1, 1), (1, -1)],
    #         noise=0.1
    #     )
    #     ass5 = Assignment5()
    #     T = time.time()
    #     shape = ass5.fit_shape(sample=polygon, maxtime=30)
    #     T = time.time() - T
    #     a = shape.area()
    #     print("the area is: " + str(a))
    #     print("the expected area is: " + str(polygon.area()))
    #     self.assertLess(abs(a - polygon.area()), 0.01)
    #     self.assertLessEqual(T, 32)
    #
    # def test_polygon_hard_case_area(self):
    #     circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
    #     ass5 = Assignment5()
    #     T = time.time()
    #     shape = ass5.fit_shape(sample=circ, maxtime=30)
    #     T = time.time() - T
    #     a = shape.area()
    #     print("the area is: " + str(a))
    #     print("the expected area is: " + str(np.pi))
    #     self.assertLess(abs(a - np.pi), 0.01)
    #     self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
