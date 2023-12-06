import os
from typing import Any, Dict, List

import click
import yaml

from divrec.utils import get_logger
from divrec_pipelines.pipeline import Stage


class Pipeline:
    def __init__(self, stages: List[Stage]):
        self.stages = stages

    def run(self):
        @click.command()
        @click.argument("config_path")
        def wrapper(config_path) -> None:
            with open(os.path.abspath(config_path), mode="r") as file:
                config: Dict[str, Any] = yaml.safe_load(file)

            if "logfile" in config["meta"]:
                logger = get_logger(
                    self.__class__.__name__, filepath=config["meta"]["logfile"]
                )
            else:
                logger = get_logger(self.__class__.__name__)

            head, tail = self.stages[0], self.stages[1:]

            head.config = config["stages"].get(head.name, dict())
            logger.info("Start stage " + head.name)
            output = head()
            logger.info("Finish stage " + head.name)
            for stage in tail:
                stage.config = config["stages"].get(stage.name, dict())
                logger.info("Start stage " + stage.name)
                output = stage(output)
                logger.info("Finish stage " + stage.name)

        return wrapper()
