import logging

FORMATTER = u'%(asctime)s - %(name)s - %(levelname)s - %(filename)s.%(funcName)s - %(message)s'


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs) -> logging.Logger:
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls].logger


class WBDLogger(metaclass=Singleton):
    logger = None

    def __init__(self, level=logging.INFO,
                 formatter=FORMATTER,
                 handler=logging.FileHandler('spam.log')):
        import sys
        self.logger = logging.getLogger('WBDLogger')
        self.logger.setLevel(level)

        if handler is None:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(level)

        handler.setFormatter(logging.Formatter(formatter, datefmt='%Y-%m-%d %H:%M:%S'))
        self.logger.addHandler(handler)

    # def info(self, msg, *args):
    #     self.logger.info(msg)
    #
    #     if args:
    #         self.logger.info('\n'.join(tuple(map(str, args))))
    #
    # def debug(self, msg, *args):
    #     self.logger.debug(msg)
    #
    #     if args:
    #         self.logger.info('\n'.join(tuple(map(str, args))))
    #
    # def warning(self, msg, *args):
    #     self.logger.warning(msg)
    #
    #     if args:
    #         self.logger.info('\n'.join(tuple(map(str, args))))
    #
    # def error(self, msg, *args):
    #     self.logger.error(msg)
    #
    #     if args:
    #         self.logger.info('\n'.join(tuple(map(str, args))))
