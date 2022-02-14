import logging


class Logger(object):
    # Закрыть доступ на переопределение init
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Logger, cls).__new__(cls)
        return cls.instance

    def __init__(self, level=logging.INFO,
                 formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                 handler=None):
        import sys

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
