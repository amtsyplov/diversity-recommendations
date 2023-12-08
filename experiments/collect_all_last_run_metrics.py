import os
import click
from typing import List

import pandas as pd
from divrec_experiments.utils import load_yaml


COLUMNS = [
    "timestamp",
    "value",
    "stage",
    "metric",
    "experiment",
    "start_time",
    "end_time",
]


def find_experiments(root: str) -> List[str]:
    experiments = []
    for folder in os.listdir(root):
        if len(folder) > 2 and folder[0].isdigit():
            experiments.append(folder)
    return experiments


def find_last_run(experiment_path: str) -> str:
    last_run_id, last_run_end_time = 0, 0
    for run in os.listdir(experiment_path):
        if run.endswith("yaml"):
            continue

        run_meta = load_yaml(os.path.join(experiment_path, run, "meta.yaml"))
        if run_meta["end_time"] > last_run_end_time:
            last_run_id = run
            last_run_end_time = run_meta["end_time"]

    return last_run_id


def collect_run_metrics(run_path: str, experiment_meta: dict) -> pd.DataFrame:
    run_meta = load_yaml(os.path.join(run_path, "meta.yaml"))
    values = pd.DataFrame(columns=COLUMNS)
    for metric in os.listdir(os.path.join(run_path, "metrics")):
        metric_values = pd.read_csv(
            os.path.join(run_path, "metrics", metric),
            sep=" ",
            names=["timestamp", "value", "stage"]
        )
        metric_values["metric"] = [metric for _ in range(len(metric_values))]
        metric_values["experiment"] = [experiment_meta["name"] for _ in range(len(metric_values))]
        metric_values["start_time"] = [run_meta["start_time"] for _ in range(len(metric_values))]
        metric_values["end_time"] = [run_meta["end_time"] for _ in range(len(metric_values))]
        values = values.append(metric_values)
    return values


@click.command()
@click.argument("root_path")
@click.argument("save_path")
def main(root_path: str, save_path: str) -> None:
    root = os.path.abspath(root_path)
    experiments = find_experiments(root)
    values = pd.DataFrame(columns=COLUMNS)
    for experiment in experiments:
        experiment_path = os.path.join(root, experiment)
        experiment_meta = load_yaml(os.path.join(root, experiment, "meta.yaml"))
        run = find_last_run(experiment_path)
        run_path = os.path.join(experiment_path, run)
        run_values = collect_run_metrics(run_path, experiment_meta)
        values = values.append(run_values)

    values.to_csv(save_path, index=False)


if __name__ == '__main__':
    main()
