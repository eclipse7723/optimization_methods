from utils.Logger import Logger


class DSKPowell:
    """
        Одномерный метод поиска 'ДСК-Пауэлл'

        dx = x3-x2 = x2-x1
        x* = x2 + ( dx*(f(x1)-f(x3)) ) / ( 2*(f(x1)-2*f(x2)+f(x3) )
    """

    MAX_ITERATIONS = 500

    def __init__(self, fn, a, b, eps=0.001):
        """
            :param a: left bound
            :param b: right bound
        """
        self.f = fn
        self.x1 = a
        self.fx1 = self.f(self.x1)
        self.x2 = (a+b)/2   # <-- central point
        self.fx2 = self.f(self.x2)
        self.x3 = b
        self.fx3 = self.f(self.x3)
        self.eps = eps

        self.iterations = 0
        self.x = None       # <--- this x* we will find

        with Logger("DSK Powell"):
            self.find_x()

    def should_stop(self, x, fx):
        """
            Criterias:
                |f(x2) - f(x*)| <= eps
                |x2 - x*| <= eps
            :return: True if criteria is passed or reached max_recursion_depth, otherwise False
        """
        if self.x is not None:
            return True
        if all([abs(self.fx2 - fx) <= self.eps,
                abs(self.x2 - x) <= self.eps]) is True:
            return True
        if self.iterations >= self.MAX_ITERATIONS:
            Logger.log("! MAX_ITERATIONS reached !")
            return True
        return False

    def find_x(self):
        """
            dx = x3-x2 = x2-x1
            x* = x2 + ( dx*(f(x1)-f(x3)) ) / ( 2*(f(x1)-2*f(x2)+f(x3) )
        """

        headers = ["i", "x1", "x2", "x3", "f(x1)", "f(x2)", "f(x3)", "x*", "f(x*)"]
        log_pattern = "{!s:^3}\t" + "{!s:<24.24}\t" * (len(headers)-1)
        Logger.log(log_pattern.format(*headers))

        while True:
            self.iterations += 1

            method = self._dsk if self.iterations == 1 else self._powell
            x, fx = method()

            Logger.log(log_pattern.format(self.iterations, self.x1, self.x2, self.x3, self.fx1, self.fx2, self.fx3, x, fx))

            if self.should_stop(x, fx) is True:
                self.x = x
                break

            if self.update_xs(x, fx) is False:
                break

        self._report()
        return self.x

    # utils

    def _report(self):
        fx = self.f(self.x)
        Logger.log(f"---> found x={self.x:.24f} (f(x)={fx:.24f}) on i={self.iterations}", new_line=True)

    def _dsk(self):
        """ first iteration (method DSK) """
        dx = self.x3 - self.x2
        _dx = self.x2 - self.x1
        Logger.assertion(bool(dx == _dx), f"dx = x3-x2 = x2-x1, but you has {dx:.48f} != {_dx:.48f}")

        x = self.x2 + dx*(self.fx1-self.fx3) / (2*(self.fx1-2*self.fx2+self.fx3))
        fx = self.f(x)

        return x, fx

    def _powell(self):
        """ another iterations (method Powell) """
        a0 = self.fx1
        a1 = (self.fx2 - self.fx1)/(self.x2 - self.x1)
        a2 = 1/(self.x3-self.x2) * ((self.fx3-self.fx1)/(self.x3-self.x1) - (self.fx2-self.fx1)/(self.x2-self.x1))

        g = lambda x: a0 + a1*(x-self.x1) + a2*(x-self.x1)*(x-self.x2)

        x = (self.x1+self.x2)/2 - a1/a2 * 0.5
        fx = g(x)

        return x, fx

    def update_xs(self, x, fx):
        # new x2 is x with smallest f(x) value:
        pairs = sorted([(self.x1, self.fx1), (self.x2, self.fx2), (self.x3, self.fx3), (x, fx)])
        self.x2, self.fx2 = min(pairs, key=lambda i: i[1])

        # and x1, x3 will be nearest to new x2:
        idx = pairs.index((self.x2, self.fx2))

        if idx == 0 or idx == len(pairs)-1:
            self.x = pairs[idx][0]
            return False

        self.x1, self.fx1 = pairs[idx - 1]
        self.x3, self.fx3 = pairs[idx + 1]
        return True

