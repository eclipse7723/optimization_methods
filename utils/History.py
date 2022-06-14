class History:
    class IterUnit:
        def __init__(self, i, x, fx, direction):
            self.i = i
            self.x = x
            self.fx = fx
            self.direction = direction

        def __repr__(self):
            return f"<i={self.i}: x={self.x}, f(x)={self.fx:.10f}>"

    def __init__(self):
        self.__values = []

    @property
    def current(self):
        if len(self.__values) == 0:
            return None
        return self.__values[-1]

    @property
    def last(self):
        if len(self.__values) < 2:
            return None
        return self.__values[-2]

    def append(self, i, x, fx, direction):
        iter_unit = History.IterUnit(i, x, fx, direction)
        self.__values.append(iter_unit)

    def clear(self):
        self.__values = []

    def items(self):
        return self.__values

    def __str__(self):
        return f"<History [{len(self.__values)} records]>"

    def __len__(self):
        return len(self.__values)

    def __getitem__(self, i):
        return self.__values[i]

    def __setitem__(self, i, value):
        if isinstance(value, (list, tuple)) is False:
            raise TypeError("Value must be list or tuple and has 3 items: i, x, fx, direction")
        if len(value) != 4:
            raise ValueError("You should input 3 values: i, x, fx, direction")
        iter_unit = History.IterUnit(*value)
        self.__values[i] = iter_unit


