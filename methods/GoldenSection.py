from utils.Logger import Logger


class GoldenSection:
    """
        Одномерный метод поиска 'Золотое сечение'
        x1 = a + 0.382*L,   x2 = a + 0.618*L
    """

    MAX_RECURSION_DEPTH = 50
    X1_COEFFICIENT = 0.382
    X2_COEFFICIENT = 0.618

    def __init__(self, fn, a, b, eps=0.001):
        self.f = fn
        self.a = a
        self.b = b

        self.iterations = 0
        self.interval = None
        self.x = None
        self.eps = eps

        with Logger("Golden Section"):
            self.set_interval()

    # main

    def should_stop(self):
        """
            Criteria: |b-a| <= eps
            :return: True if criteria is passed or reached max_recursion_depth, otherwise False
        """
        if abs(self.b - self.a) <= self.eps:
            return True
        if self.iterations > self.MAX_RECURSION_DEPTH:
            Logger.log("! MAX_RECURSION_DEPTH reached !")
            return True
        return False

    def set_interval(self):
        X1, X2 = self.X1_COEFFICIENT, self.X2_COEFFICIENT

        headers = ["i", "a", "x1", "x2", "b", "L", "f(x1)", "f(x2)"]
        log_pattern = "{!s:<12.12}\t" * len(headers)
        Logger.log(log_pattern.format(*headers))

        def recursion_finder(i=1):
            if self.should_stop() is True:
                interval = [self.a, self.b]
                return interval

            self.iterations = i

            L = self.b - self.a
            x1 = self.a + X1 * L
            x2 = self.a + X2 * L
            fx1 = self.f(x1)
            fx2 = self.f(x2)

            Logger.log(log_pattern.format(i, self.a, x1, x2, self.b, L, fx1, fx2))

            if fx1 <= fx2:
                self.b = x2
            else:
                self.a = x1

            return recursion_finder(i+1)

        self.interval = recursion_finder()
        self.x = sum(self.interval) / 2

        Logger.log(f"---> found x={self.x} and interval={self.interval} on i={self.iterations}", new_line=True)




