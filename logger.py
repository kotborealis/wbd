import logging


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        print("call")
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    def __init__(self, level=logging.INFO,
                 formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                 handler=None):
        import sys
        print("init")
        self.logger = logging.getLogger('WBDLogger')
        self.logger.setLevel(level)

        if handler is None:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(level)

        handler.setFormatter(logging.Formatter(formatter))
        self.logger.addHandler(handler)

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

