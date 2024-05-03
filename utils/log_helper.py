from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
from datetime import datetime


def init_logger(log_file):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def pre_log(Logger, args):
    Logger.info('\nTraining date: %s', datetime.now())
    for arg in vars(args):
        Logger.info(f"{arg}: {getattr(args, arg)}")