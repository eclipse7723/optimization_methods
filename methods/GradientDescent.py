""" Метод наискорейшего спуска """


from utils.Logger import Logger
from utils.utils import gradient, get_vector_norm
from utils.History import History

from methods.Sven import Sven
from methods.GoldenSection import GoldenSection
from methods.DSKPowell import DSKPowell


class GradientDescentMixin:

    """
        x_(k+1) = x_k - step_k * grad(x_k)
    """

    TYPE = None
    MODIFICATION = None
    MAX_RECURSION_DEPTH = 200

    def __init__(self, fn, start_point, step, eps=0.001, grad=None, **kwargs):
        """ :param grad: should be a function that takes one arg: np.ndarray """

        self.iterations = 0
        self.history = History()

        self.f = fn
        self.start_point = start_point.astype(float)
        self.eps = eps
        self.step = step

        if grad is None:
            # numeric way
            self.grad = lambda point: gradient(fn, point)
        else:
            # your own gradient function
            if callable(grad) is False:
                raise TypeError("Your gradient is not callable, also it should return np.ndarray")
            self.grad = grad

    def start(self):
        title = f"Gradient Descent ({self.TYPE})"
        title += f", mod: {self.MODIFICATION}" if self.MODIFICATION is not None else ""
        with Logger(title):
            self.find_x()
        return self

    def update_step(self):
        raise Exception("This class is Mixin, you should use class `GradientDescent`")

    def find_step(self, input_step):
        """ set first step """
        raise Exception("This class is Mixin, you should use class `GradientDescent`")

    def get_direction(self, gx, norm):
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
        log_pattern = "{!s:^3}\t" + "{!s:<35.35}\t" * (len(headers)-1)
        Logger.log(log_pattern.format(*headers))

        def recursive_finder(x, i=0):
            self.iterations = i
            fx = self.f(*x)
            gx = self.grad(x)
            norm = get_vector_norm(gx)
            direction = self.get_direction(gx, norm)

            Logger.log(log_pattern.format(i, x, fx, gx, norm, direction, self.step))
            self.history.append(i, x, fx, direction)

            if self.should_stop(norm) is True:
                Logger.log(f"---> found x=({x[0]:.24f}, {x[1]:.24f}) on i={self.iterations}", new_line=True)
                return x

            self.update_step()
            next_x = self.get_next_x(x, self.step, direction)

            return recursive_finder(next_x, i=i+1)

        self.x = recursive_finder(self.start_point)
        return self.x


# Main algorithms:


class OptimalGradientDescent(GradientDescentMixin):
    TYPE = "optimal"
    SVEN_STEP = None
    ONE_DIM_METHOD = DSKPowell

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.g = lambda step: self.f(*(self.history.current.x + step * self.history.current.direction))

    def find_step(self, input_step=None):
        current = self.history.current
        sven_step = self.SVEN_STEP or 0.1 * get_vector_norm(current.x) / get_vector_norm(current.direction)
        interval = Sven(self.g, 0, sven_step).interval
        step = self.ONE_DIM_METHOD(self.g, *interval).x
        return step

    def get_direction(self, gx, norm):
        direction = -gx
        return direction

    def update_step(self):
        self.step = self.find_step()


class ConstGradientDescent(GradientDescentMixin):
    TYPE = "const"

    def get_direction(self, gx, norm):
        direction = -gx/norm
        return direction

    def update_step(self):
        if self.history.last is None:
            return
        if self.history.current.fx > self.history.last.fx:
            self.step /= 2


# Modifications:


class BoothGradientDescent(OptimalGradientDescent):
    MODIFICATION = "Booth"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = kwargs.get("delta", 0.8)

    def get_next_x(self, x, step, direction):
        return x + self.delta * step * direction


# Factory:


class GradientDescent:
    MODIFICATIONS = {
        "booth": BoothGradientDescent
    }

    @classmethod
    def optimal_setup_params(cls, params):
        one_dim_method = params.get("one_dim_method", "dsk_powell").lower()
        if one_dim_method == "dsk_powell":
            OptimalGradientDescent.ONE_DIM_METHOD = DSKPowell
        elif one_dim_method == "golden_section":
            OptimalGradientDescent.ONE_DIM_METHOD = GoldenSection

        if "sven_step" in params:
            OptimalGradientDescent.SVEN_STEP = float(params["sven_step"])

    @classmethod
    def choose_modification(cls, params):
        if "modification" not in params:
            return

        mod = params["modification"].lower()
        mod_gradient_descent = cls.MODIFICATIONS.get(mod)
        return mod_gradient_descent

    def __new__(cls, fn, start_point, step=None, eps=0.001, grad=None, **params):
        if step is None:
            cls.optimal_setup_params(params)

        mod_gradient_descent = cls.choose_modification(params)
        if mod_gradient_descent is not None:
            return mod_gradient_descent(fn, start_point, step, eps, grad, **params)

        if step is None:
            return OptimalGradientDescent(fn, start_point, step, eps, grad)
        else:
            return ConstGradientDescent(fn, start_point, step, eps, grad)





