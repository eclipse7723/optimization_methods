import numpy as np
from methods.Sven import Sven
from methods.GoldenSection import GoldenSection
from methods.DSKPowell import DSKPowell
from methods.GradientDescent import GradientDescent


def rgr_first_iter_test():
    print(" >>> RGR i=3 test:")

    fn = lambda x1, x2: 6*(x1-9)**2 + x1*x2 + 2*x2**2
    x0 = np.array([16.7, 16.7])
    g = lambda step: fn(*(x0 + step*np.array([0, 1])))
    step = 2.361736649163069
    eps = 0.001

    interval = Sven(g, 0, step).interval
    step = GoldenSection(g, *interval, eps).x


def rgr_third_iter_test():
    print(" >>> RGR i=3 test:")

    fn = lambda x1, x2: 6*(x1-9)**2 + x1*x2 + 2*x2**2
    x0 = np.array([9.34781451, -4.17495213])
    g = lambda step: fn(*(x0 + step*np.array([0, 1])))
    step = 1.0237766424932546
    eps = 0.001

    interval = Sven(g, 0, step).interval
    step = DSKPowell(g, *interval, eps).x


def gradient_descent_const_test():
    fn = lambda x1, x2: 3*x1**2 + x1*x2 + 2*x2**2
    gradient = lambda x1, x2: np.array([6*x1+x2, x1+4*x2])
    x0 = np.array([6.0, 4.0])
    step = 3

    result = GradientDescent(fn, x0, step=step, grad=lambda point: gradient(*point)).start()
    return result


def gradient_descent_optimal_test():
    fn = lambda x1, x2: 4*x1**2 + x1*x2 + x2**2
    gradient = lambda x1, x2: np.array([8*x1+x2, x1+2*x2])
    grad = None # lambda point: gradient(*point)
    x0 = np.array([6, 4])

    params = {
        "one_dim_method": "dsk_powell"
    }

    result = GradientDescent(fn, x0, grad=grad, **params).start()
    return result
