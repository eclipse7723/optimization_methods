""" Метод наискорейшего спуска """


from utils.Logger import Logger
from utils.utils import gradient, get_vector_norm
from utils.History import History


class GradientDescentMixin:

    """
        x_(k+1) = x_k - step_k * grad(x_k)
    """

    TYPE = None
    MAX_RECURSION_DEPTH = 200

    def __init__(self, fn, start_point, step, eps=0.001, grad=None):
        """ :param grad: should be a function that takes one arg: np.ndarray """

        self.iterations = 0
        self.f = fn
        self.start_point = start_point
        self.step = self.find_step(step)
        self.eps = eps

        if grad is None:
            # numeric way
            self.grad = lambda point: gradient(fn, point)
        else:
            # your own gradient function
            if callable(grad) is False:
                raise TypeError("Your gradient is not callable, also it should return np.ndarray")
            self.grad = grad

        self.history = History()

        with Logger(f"Gradient Descent ({self.TYPE})"):
            self.x = self.find_x()

    def update_step(self):
        raise Exception("This class is Mixin, you should use class `GradientDescent`")

    def find_step(self, input_step):
        """ set first step """
        raise Exception("This class is Mixin, you should use class `GradientDescent`")

    def get_next_x(self, x, step, direction):
        return x + step * direction

    def should_stop(self, norm):
        if norm <= self.eps:
            return True
        if self.iterations >= self.MAX_RECURSION_DEPTH:
            Logger.log("! MAX_RECURSION_DEPTH reached !")
            return True
        return False

    def find_x(self):
        headers = ["i", "x", "f(x)", "∇f(x)", "||∇f(x)||", "direction", "step"]
        log_pattern = "{!s:^3}\t" + "{!s:<25.25}\t" * (len(headers)-1)
        Logger.log(log_pattern.format(*headers))

        def recursive_finder(x, i=0):
            self.iterations = i
            fx = self.f(*x)
            gx = self.grad(x)
            norm = get_vector_norm(gx)
            direction = -gx/norm

            Logger.log(log_pattern.format(i, x, fx, gx, norm, direction, self.step))
            self.history.append(i, x, fx)

            if self.should_stop(norm) is True:
                Logger.log(f"\n---> found x={x} on i={self.iterations}")
                return x

            self.update_step()
            next_x = self.get_next_x(x, self.step, direction)

            return recursive_finder(next_x, i=i+1)

        self.x = recursive_finder(self.start_point)
        return self.x


class OptimalGradientDescent(GradientDescentMixin):
    TYPE = "optimal"
    # todo


class ConstGradientDescent(GradientDescentMixin):
    TYPE = "const"

    def find_step(self, input_step):
        return input_step

    def update_step(self):
        if self.history.last is None:
            return
        if self.history.current.fx > self.history.last.fx:
            self.step /= 2


class GradientDescent:
    def __new__(cls, fn, start_point, step=None, eps=0.001, grad=None):
        if step is None:
            return OptimalGradientDescent(fn, start_point, step, eps, grad)
        else:
            return ConstGradientDescent(fn, start_point, step, eps, grad)





