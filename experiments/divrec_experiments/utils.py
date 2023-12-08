import os
import json
import logging
from typing import Any, Dict, Optional
import random
import numpy as np
import torch
import yaml


LOG_FORMAT = f"%(asctime)s %(name)s [%(levelname)s] %(message)s"


def to_json(data, filepath):
    with open(filepath, mode="w") as file:
        json.dump(data, file)


def load_yaml(filepath: str) -> Dict[str, Any]:
    with open(filepath, mode="r") as file:
        return yaml.safe_load(file)


def create_if_not_exist(filepath: str) -> None:
    if not os.path.exists(filepath):
        os.mkdir(filepath)


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


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
