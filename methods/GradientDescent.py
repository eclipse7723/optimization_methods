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
    MAX_ITERATIONS = 2000

    def __init__(self, fn, start_point, step, grad=None, **params):
        """ :param grad: should be a function that takes one arg: np.ndarray """

        self.iterations = 0
        self.history = History()

        self.x = None
        self.f = fn
        self.start_point = start_point.astype(float)
        self.step = step

        if grad is None:
            # numeric way
            self.grad = lambda point: gradient(fn, point)
        else:
            # your own gradient function
            if callable(grad) is False:
                raise TypeError("Your gradient is not callable, also it should return np.ndarray")
            self.grad = grad

        self.criteria_eps = params.get("criteria_eps", 10**-3)
        self.criteria = params.get("criteria", 1)
        # 0 - ||x_(k+1) - x_k|| / ||x_k|| < eps AND |f(x_(k+1) - f(x_k)| < eps
        # 1 - ||∇f(x)|| < eps

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
        next_x = x + step * direction
        return next_x

    def _check_criteria(self, gx_norm):
        if self.criteria == 0:
            if self.history.last is None:
                return False
            cur, last = self.history.current, self.history.last
            if all([get_vector_norm(cur.x - last.x)/get_vector_norm(last.x) <= self.criteria_eps,
                    abs(cur.fx - last.fx) <= self.criteria_eps]) is True:
                return True
        elif self.criteria == 1:
            if gx_norm <= self.criteria_eps:
                return True

        return False

    def should_stop(self, gx_norm):
        if self._check_criteria(gx_norm) is True:
            return True
        if self.iterations >= self.MAX_ITERATIONS:
            Logger.log("! MAX_RECURSION_DEPTH reached !")
            return True
        return False

    def find_x(self):
        headers = ["i", "x", "f(x)", "∇f(x)", "||∇f(x)||", "direction", "step"]
        log_pattern = "{!s:^3}\t" + "{!s:<35.35}\t" * (len(headers)-1)
        Logger.log(log_pattern.format(*headers))

        x = self.start_point
        while True:
            i = self.iterations

            fx = self.f(*x)
            gx = self.grad(x)
            norm = get_vector_norm(gx)
            direction = self.get_direction(gx, norm)

            Logger.log(log_pattern.format(i, x, fx, gx, norm, direction, self.step))
            self.history.append(i, x, fx, direction)

            if self.should_stop(norm) is True:
                Logger.log(f"---> found x=({x[0]:.24f}, {x[1]:.24f}) on i={self.iterations}", new_line=True)
                self.x = x
                break

            self.update_step()
            x = self.get_next_x(x, self.step, direction)
            self.iterations += 1

        return x


# Main algorithms:


class OptimalGradientDescent(GradientDescentMixin):
    TYPE = "optimal"
    ONE_DIM_METHOD = DSKPowell

    def __init__(self, *args, **params):
        super().__init__(*args, **params)
        self.g = lambda step: self.f(*(self.history.current.x + step * self.history.current.direction))
        self.sven_step = params.get("sven_step", None)
        self.one_dim_eps = params.get("one_dim_eps", 10**-3)
        self._report_one_dim_details = params.get("report_one_dim_details", False)

    def find_step(self, input_step=None):
        logger_enable = Logger.ENABLE
        if self._report_one_dim_details is False:
            Logger.ENABLE = False

        current = self.history.current
        sven_step = self.sven_step or 0.1 * get_vector_norm(current.x) / get_vector_norm(current.direction)
        interval = Sven(self.g, 0, sven_step).interval
        step = self.ONE_DIM_METHOD(self.g, *interval, eps=self.one_dim_eps).x

        if self._report_one_dim_details is False:
            Logger.ENABLE = logger_enable
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
            self.history.pop()
            self.step /= 2


# Modifications:


class BoothGradientDescent(OptimalGradientDescent):
    MODIFICATION = "Booth"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = kwargs.get("delta", 0.8)

    def get_next_x(self, x, step, direction):
        return x + self.delta * step * direction


class HeavyBallGradientDescent(OptimalGradientDescent):
    # fixme
    MODIFICATION = "HeavyBall"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = kwargs.get("delta", 0.8)

    def get_next_x(self, x, step, direction):
        if self.history.last is None:
            heavy = 0
        else:
            heavy = self.delta * (self.history.last.x - x)
        next_x = x + step * direction - heavy
        return next_x


class LyusternikGradientDescent(OptimalGradientDescent):
    # fixme
    MODIFICATION = "Lyusternik"

    def get_beta_k(self):
        if self.history.last is None:
            return
        cur, last = self.history.current, self.history.last
        norm_cur = get_vector_norm(self.grad(cur.x))
        norm_last = get_vector_norm(self.grad(last.x))
        beta_k = norm_cur / norm_last
        return beta_k

    def update_step(self):
        if self.history.last is None:
            self.step = self.find_step()
            return

        beta_k = self.get_beta_k()
        beta_coef = (beta_k / (1.0-beta_k))
        self.step = beta_coef


# Factory:


class GradientDescent:
    MODIFICATIONS = {
        "booth": BoothGradientDescent,
        "heavy_ball": HeavyBallGradientDescent,
        "lyusternik": LyusternikGradientDescent,
    }
    ONE_DIM_METHODS = {
        "dsk_powell": DSKPowell,
        "golden_section": GoldenSection
    }

    @classmethod
    def optimal_setup_params(cls, params):
        one_dim_method = params.get("one_dim_method", "golden_section").lower()
        if one_dim_method in cls.ONE_DIM_METHODS:
            OptimalGradientDescent.ONE_DIM_METHOD = cls.ONE_DIM_METHODS[one_dim_method]

        if "sven_step" in params:
            OptimalGradientDescent.SVEN_STEP = float(params["sven_step"])

    @classmethod
    def choose_modification(cls, params):
        if "modification" not in params:
            return

        mod = params["modification"].lower()
        mod_gradient_descent = cls.MODIFICATIONS.get(mod)
        return mod_gradient_descent

    def __new__(cls, fn, start_point, step=None, grad=None, **params):
        if step is None:
            cls.optimal_setup_params(params)

        mod_gradient_descent = cls.choose_modification(params)
        if mod_gradient_descent is not None:
            return mod_gradient_descent(fn, start_point, step=step, grad=grad, **params)

        if step is None:
            return OptimalGradientDescent(fn, start_point, step=step, grad=grad, **params)
        else:
            return ConstGradientDescent(fn, start_point, step=step, grad=grad, **params)





