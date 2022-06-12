import time
import numpy as np


def benchmark(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        return_value = func(*args, **kwargs)
        end = time.time()
        print('[*] Runtime: {} s.'.format(end-start))
        return return_value
    return wrapper


def gradient(fn, point, h=0.00001):
    """ central differences method

        :type fn: function
        :type point: float|np.ndarray
    """
    if isinstance(point, np.ndarray):
        point = point.astype(float)
        dxs = []
        for i in range(len(point)):
            p = point.copy()
            p[i] += h
            dx = (fn(*p) - fn(*point))/h
            dxs.append(dx)
        return np.array(dxs)

    return (fn(point+h) - fn(point-h))/(2*h)


def get_vector_norm(vec: np.ndarray):
    norm = np.sqrt(sum(vec**2))
    return norm


def get_vector_direction(vec: np.ndarray):
    norm = get_vector_norm(vec)
    direction = vec/norm
    return direction


class Affector:
    """
    Example: increase h in gradient for each call
    >>> f = lambda x1, x2: x2**2+x1*x2-x1**5
    >>> wrapper = lambda point, h: gradient(f, point, h=10**-h)
    >>> grad = Affector(wrapper, "h")
    >>> grad(np.array([1.0, 5.0]))
    """

    def __init__(self, fn, key):
        self.fn = fn
        self.calls = 0
        self.key = key

    def __call__(self, *args, **kwargs):
        kwargs[self.key] = self.calls
        result = self.fn(*args, **kwargs)
        self.calls += 1
        return result
