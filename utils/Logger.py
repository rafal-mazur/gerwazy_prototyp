from enum import Enum


class Color(Enum):
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


class Logger:
    def __init__(self, enable_logging: bool = True) -> None:
        self._enable_logging: bool = enable_logging

    def set_logging(self, enable_logging: bool) -> None:
        self._enable_logging = enable_logging
        
    def done(self) -> None:
        if self._enable_logging:
            print(Color.GREEN.value + 'Done' + Color.ENDC.value, end='\n\n')
    

    def __call__(self, *args, color: Color|None = None):
        if self._enable_logging:
            if color is None:
                print(*args)
            else:
                for i in args:
                    print(color.value + i, end=Color.ENDC.value + ' ')
                print()
