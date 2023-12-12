import os
import click
from typing import List
from dataclasses import dataclass

import pandas as pd
from divrec_experiments.utils import load_yaml, get_logger, create_if_not_exist, get_workdir


COLUMNS = [
    "timestamp",
    "value",
    "stage",
    "metric",
    "experiment",
    "start_time",
    "end_time",
]


@dataclass
class Run:
    id: str
    start_time: int
    end_time: int

    def __gt__(self, other):
        return self.end_time > other.end_time


def find_experiments(root: str) -> List[str]:
    experiments = []
    for folder in os.listdir(root):
        if folder[0].isdigit():
            experiments.append(folder)
    return experiments


def find_last_run(experiment_path: str) -> Run:
    def load(idx: str) -> Run:
        run_meta = load_yaml(os.path.join(experiment_path, idx, "meta.yaml"))
        return Run(idx, run_meta["start_time"], run_meta["end_time"])

    last_run = Run("0", 0, 0)
    for run_id in os.listdir(experiment_path):
        if run_id.endswith("yaml"):
            continue

        run = load(run_id)
        if run > last_run:
            last_run = run

    return last_run


def collect_run_metrics(experiment_path: str, run: Run) -> pd.DataFrame:
    experiment_name = load_yaml(os.path.join(experiment_path, "meta.yaml"))["name"]
    values = pd.DataFrame(columns=COLUMNS)
    for metric in os.listdir(os.path.join(experiment_path, run.id, "metrics")):
        metric_values = pd.read_csv(
            os.path.join(experiment_path, run.id, "metrics", metric),
            sep=" ",
            names=["timestamp", "value", "stage"]
        )
        metric_values["metric"] = [metric for _ in range(len(metric_values))]
        metric_values["experiment"] = [experiment_name for _ in range(len(metric_values))]
        metric_values["start_time"] = [run.start_time for _ in range(len(metric_values))]
        metric_values["end_time"] = [run.end_time for _ in range(len(metric_values))]
        values = pd.concat((values, metric_values), ignore_index=True)
    return values


@click.command()
@click.option("-s", "--source", "source", default=".mlruns", type=str)
@click.option("-t", "--target", "target", default="runs.csv", type=str)
def main(source: str, target: str) -> None:
    logger = get_logger(__file__)
    workdir = get_workdir({}, __file__)
    create_if_not_exist(workdir)
    root = os.path.abspath(source)
    experiments = find_experiments(root)
    logger.info(f"Found experiments: {', '.join(experiments)}")
    values = pd.DataFrame(columns=COLUMNS)
    for experiment in experiments:
        experiment_path = os.path.join(root, experiment)
        run = find_last_run(experiment_path)
        try:
            run_values = collect_run_metrics(experiment_path, run)
            values = pd.concat((values, run_values), ignore_index=True)
            logger.info(f"Finish processing \"{experiment}/{run.id}\"")
        except FileNotFoundError:
            logger.warning(f"There are no metrics in {experiment_path}/{run.id}/")
    values.to_csv(os.path.join(workdir, target), index=False)


if __name__ == '__main__':
    main()
