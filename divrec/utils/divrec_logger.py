import logging
from typing import Optional

LOG_FORMAT = f"%(asctime)s %(name)s [%(levelname)s] %(message)s"


def get_file_handler(filepath: str):
    file_handler = logging.FileHandler(filepath)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return file_handler


def get_stream_handler():
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return stream_handler


def get_logger(name: str, filepath: Optional[str] = None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if filepath is not None:
        logger.addHandler(get_file_handler(filepath))
    logger.addHandler(get_stream_handler())
    return logger
