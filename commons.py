from math import e, log
from numpy import sin
from numpy.ma import arctan
from numpy import power as pow, empty, float64

from functionUtils import *
from sampleFunctions import *


def f1(x):
    if type(x) is float64:
        return float64(5)
    if type(x) is int:
        return 5
    if type(x) is float:
        return 5.0
    a = empty(len(x))
    a.fill(5)
    return a


def f11(x):
    return 5.0


def f12(x):
    return 5.0


def f13(x):
    return 5.0


def f14(x):
    return 5.0


def f2(x):
    return pow(x, 2) - 3 * x + 2


def f25(x):
    return pow(x, 2) - 3 * x + 2


def f23(x):
    return pow(x, 2) - 3 * x + 2


def f27(x):
    return pow(x, 2) - 3 * x + 2


def f2_nr(x):
    return pow(x, 2) - 3 * x + 2


@NOISY(3)
def f2_noise(x):
    return pow(x, 2) - 3 * x + 2


def f3(x):
    return sin(pow(x, 2))


def f32(x):
    return sin(pow(x, 2))


def f3_nr(x):
    return sin(pow(x, 2))


@NOISY(1)
def f3_noise(x):
    return sin(pow(x, 2))


def f6(x):
    return sin(x) / x - 0.1


def f9(x):
    return log(log(x))


@NOISY(1)
def f9_noise(x):
    return log(log(x))


def f10(x):
    return sin(log(x))


class Polygon(AbstractShape):
    def __init__(self, knots, noise):
        self._knots = knots
        self._noise = noise
        self._n = len(knots)

    def sample(self):
        i = np.random.randint(self._n)
        t = np.random.random()

        x1, y1 = self._knots[i - 1]
        x2, y2 = self._knots[i]

        x = np.random.random() * (x2 - x1) + x1
        x += np.random.randn() * self._noise

        y = np.random.random() * (y2 - y1) + y1
        y += np.random.randn() * self._noise
        return x, y

    def contour(self, n: int):
        ppf = n // self._n
        rem = n % self._n
        points = []
        for i in range(self._n):
            ts = np.linspace(0, 1, num=(ppf + 2 if i < rem else ppf + 1))

            x1, y1 = self._knots[i - 1]
            x2, y2 = self._knots[i]

            for t in ts[0:-1]:
                x = t * (x2 - x1) + x1
                y = t * (y2 - y1) + y1
                xy = np.array((x, y))
                points.append(xy)
        points = np.stack(points, axis=0)
        return points

    def area(self):
        a = 0
        for i in range(self._n):
            x1, y1 = self._knots[i - 1]
            x2, y2 = self._knots[i]
            a += 0.5 * (x2 - x1) * (y1 + y2)
        return a


class BezierShape(AbstractShape):
    def __init__(self, knots, control, noise):
        self._knots = knots
        self._control = control
        self._noise = noise
        self._n = len(knots)

        self._fs = [
            bezier3(knots[i - 1], control[2 * i], control[2 * i + 1], knots[i])
            for i in range(self._n)
        ]

    def sample(self):
        i = np.random.randint(self._n)
        t = np.random.random()
        x, y = self._fs[i](t)
        x += np.random.randn() * self._noise
        y += np.random.randn() * self._noise
        return x, y

    def contour(self, n: int):
        ppf = n // self._n
        rem = n % self._n
        points = []
        for i in range(self._n):
            ts = np.linspace(0, 1, num=(ppf + 2 if i < rem else ppf + 1))
            for t in ts[0:-1]:
                x, y = self._fs[i](t)
                xy = np.array((x, y))
                points.append(xy)
        points = np.stack(points, axis=0)
        return points

    def area(self):
        a = 0
        cntr = self.contour(10000)
        for i in range(10000):
            x1, y1 = cntr[i - 1]
            x2, y2 = cntr[i]
            if x1 != x2:
                a += 0.5 * (x2 - x1) * (y1 + y2)
        return a


def noisy_circle(cx, cy, radius, noise) -> AbstractShape:
    return Circle(cx, cy, radius, noise).sample


def shape1() -> AbstractShape:
    return Polygon(
        knots=[(0, 0), (1, 1), (1, -1)],
        noise=0.1
    )


def shape3() -> AbstractShape:
    return Polygon(
        knots=[(0, 0), (0, 8), (20, 1), (1, 0)],
        noise=0.3
    )


def shape5() -> AbstractShape:
    return Polygon(
        knots=[(0, 0), (0.5, 100000), (1, 1), (100000, 1), (50000, 0)],
        noise=1.0
    )


def k1(x):
    return np.exp(-2 * (x ** 2))


def k2(x):
    return np.arctan(x)


@NOISY(0.5)
def k2Noise(x):
    return np.arctan(x)


def k3(x):
    return np.exp(np.exp(x))


def k4(x):
    return 2 ** (1 / (x ** 2)) * sin(1 / x)


def k5(x):
    return 1 / sin(x)


def k6(x):
    return sin(x) * np.cos(x)


def k61(x):
    return sin(x) * np.cos(x)


@NOISY(1)
def k6Noise(x):
    return sin(x) * np.cos(x)


def k7(x):
    return np.e ** x


def k8(x):
    return x ** 4 - x ** 3 + x ** 2


@NOISY(2)
def k8Noise(x):
    return x ** 4 - x ** 3 + x ** 2


def k9(x):
    return (x - 5) ** 2 - 6 * x + 1


@NOISY(0.5)
def k9Noise(x):
    return (x - 5) ** 2 - 6 * x + 1


if __name__ == '__main__':
    r1 = f1(3)
    r2 = f2(3)
    r3 = f3(3)
    # r4 = f4(4)
