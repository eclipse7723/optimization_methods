from utils.Logger import Logger


class Sven:
    """
        Этап установления границ интервала
        Находит интервал неопределённости
    """

    MAX_ITERATIONS = 2000

    def __init__(self, fn, start_point, step):
        self.f = fn
        self.start_point = start_point

        self.iterations = 0

        self.interval = None
        self.x = None
        self.step = None

        with Logger("Sven interval"):
            if self._set_step(step) is False:
                self.interval = [self.start_point-step, self.start_point+step]
                self.x = sum(self.interval) / 2
                self._report()
            else:
                self.set_interval()

    def _set_step(self, step):
        """
            1) f(x0-step) >= f(x0) >= (fx0+step) --> step "+"
            2) f(x0-step) <= f(x0) <= (fx0+step) --> step "-"
            3) if None --> interval is [x0-step, x0+step]
        """
        x0 = self.start_point

        fx = self.f(x0)
        fx_neg = self.f(x0 - step)
        fx_pos = self.f(x0 + step)

        if fx_neg >= fx >= fx_pos:
            self.step = step
            return True
        if fx_neg <= fx <= fx_pos:
            self.step = -step
            return True
        return False

    def should_stop(self, fx0, fx1):
        """
            Criteria: f(x0) < f(x1) - means, that on x1 function starts growing
            :return: True if criteria is passed or reached max_recursion_depth, otherwise False
        """
        max_recursion_reached = self.iterations > self.MAX_ITERATIONS
        if max_recursion_reached or fx0 < fx1:
            if max_recursion_reached:
                Logger.log("! MAX_RECURSION_DEPTH reached !")
            return True
        return False

    def set_interval(self):
        """
            formula: `x_k+1 = x_k +- step * 2^k`.

            1) f(x0-step) >= f(x0) <= (fx0+step) --> interval is [x0-step, x0+step]
            2) f(x0-step) <= f(x0) >= (fx0+step) --> no interval, return None
        """

        headers = ["k", "x_k", "∆*2^k", "x_(k+1)", "f(x_k)", "f(x_(k+1))"]
        log_pattern = "{!s:^3}\t" + "{!s:<24.24}\t" * (len(headers)-1)
        Logger.log(log_pattern.format(*headers))

        if self.step is None:
            # interval found on step setup
            return

        x0 = self.start_point
        while True:
            i = self.iterations
            step = self.step * 2**i
            x1 = x0 + step
            fx0, fx1 = self.f(x0), self.f(x1)
            Logger.log(log_pattern.format(i, x0, step, x1, fx0, fx1))

            if self.should_stop(fx0, fx1):
                self.iterations += 1    # adjust
                self.interval = sorted([x0-step/2, x0+step/2])
                break

            x0 = x1
            self.iterations += 1

        self.x = sum(self.interval) / 2
        self._report()

    def _report(self):
        Logger.log(f"---> found x={self.x} and interval=({self.interval[0]:.24f}, {self.interval[1]:.24f}) on i={self.iterations}", new_line=True)
