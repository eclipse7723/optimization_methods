class Logger:
    ENABLE = True
    WIDTH = 100
    FILLCHAR = "="
    CURRENT_DEPTH = 0

    class Colors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

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
        print(_new_line + " "*Logger.CURRENT_DEPTH*3 + f"  {msg}")

    @classmethod
    def assertion(cls, expression, msg):
        if expression is True:
            return True     # OK
        input(" "*Logger.CURRENT_DEPTH*3 + f"{Logger.Colors.FAIL}AssertionError: {msg}{Logger.Colors.ENDC}")
        exit()
        return False

    @classmethod
    def setEnable(cls, state):
        cls.ENABLE = bool(state)
