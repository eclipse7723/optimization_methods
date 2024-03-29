class Logger:
    ENABLE = True
    ASSERTION_EXIT = True

    WIDTH = 100
    FILLCHAR = "="
    CURRENT_DEPTH = 0

    class Colors:
        FAIL = '\033[91m'
        ENDC = '\033[0m'

    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        if self.ENABLE is False:
            return
        print(" "*Logger.CURRENT_DEPTH*3 + f" {self.msg} ".center(self.WIDTH, self.FILLCHAR))
        Logger.CURRENT_DEPTH += 1

    def __exit__(self, *args):
        if self.ENABLE is False:
            return
        Logger.CURRENT_DEPTH -= 1
        print(" "*Logger.CURRENT_DEPTH*3 + self.FILLCHAR*self.WIDTH)

    @classmethod
    def log(cls, msg, new_line=False):
        if cls.ENABLE is False:
            return
        _new_line = "\n" if new_line else ""
        print(_new_line + " "*Logger.CURRENT_DEPTH*3 + f"{msg}")

    @classmethod
    def assertion(cls, expression, msg):
        if expression is True:
            return True     # OK
        f_msg = " "*Logger.CURRENT_DEPTH*3 + f"{Logger.Colors.FAIL}AssertionError: {msg}{Logger.Colors.ENDC}"
        if cls.ASSERTION_EXIT is True:
            input(f_msg)
            exit()
        else:
            print(f_msg)
        return False

    @classmethod
    def setEnable(cls, state):
        cls.ENABLE = bool(state)

