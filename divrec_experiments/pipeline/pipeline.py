import os
from typing import List

import click
import yaml

from divrec_experiments.pipeline import Stage


class Pipeline:
    def __init__(self, stages: List[Stage]):
        self.stages = stages

    @click.command()
    @click.argument("config_path")
    def run(self, config_path) -> None:
        with open(os.path.abspath(config_path), mode="r") as file:
            config = yaml.safe_load(file)

        head, tail = self.stages[0], self.stages[1:]

        head.config = config.get(head.name, dict())
        output = head()
        for stage in tail:
            stage.config = config.get(stage.name, dict())
            output = stage(output)
