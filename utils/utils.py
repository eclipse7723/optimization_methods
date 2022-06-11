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


def gradient(fn, point, h=0.001):
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
