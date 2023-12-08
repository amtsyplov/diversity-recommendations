import os
from typing import Any, Dict

import yaml
import click
import mlflow

from divrec.models import RandomModel
from divrec.datasets import PairWiseDataset, RankingDataset
from divrec.metrics import (
    AUCScore,
    EntropyDiversityScore,
    MeanAveragePrecisionAtKScore,
    NDCGScore,
    PRI,
    PrecisionAtKScore,
    RecallAtKScore,
)
from divrec.train.utils import pair_wise_score_loop, recommendations_score_loop
from divrec_experiments.datasets import movie_lens_load
from divrec_experiments.utils import create_if_not_exist, get_logger, seed_everything


def load_config(path: str) -> Dict[str, Any]:
    with open(os.path.abspath(path), mode="r") as file:
        return yaml.safe_load(file)


@click.command()
@click.argument("config_path")
def main(config_path: str) -> None:
    config = load_config(config_path)
    workdir = config["workdir"] if "workdir" in config else os.path.abspath("workdir")
    create_if_not_exist(workdir)
    logger = get_logger(
        config["mlflow_experiment"],
        filepath=os.path.join(workdir, config["logfile"]),
    )
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["mlflow_experiment"])
    mlflow.log_artifact(config_path)  # save config for experiment
    seed_everything(config["seed"])

    datasets = movie_lens_load(config["data_directory"])
    logger.info("Successfully load data")

    model = RandomModel(datasets.test.number_of_users, datasets.test.number_of_items)
    model.to(config["device"])

    pair_wise_test_dataset = PairWiseDataset(
        datasets.test, frozen=datasets.train, max_sampled=20
    )

    auc_score = pair_wise_score_loop(
        pair_wise_test_dataset,
        model,
        [AUCScore()],
        batch_size=200,
    )
    logger.info(f"Successfully evaluate AUC on test: {auc_score[0].item()}")
    mlflow.log_metric("auc_score", auc_score[0].item())

    losses = [
        EntropyDiversityScore(dataset=datasets.train),
        MeanAveragePrecisionAtKScore(),
        NDCGScore(),
        PRI(dataset=datasets.train),
        PrecisionAtKScore(),
        RecallAtKScore(),
    ]
    ranking_test_dataset = RankingDataset(datasets.test, frozen=datasets.train)
    loss_values = recommendations_score_loop(
        ranking_test_dataset, model, losses, number_of_recommendations=config["k"]
    )
    for loss, value in zip(losses, loss_values):
        loss_name = f"{loss.__class__.__name__}_at_{config['k']}"
        logger.info(f"Successfully evaluate {loss_name} on test: {value.item()}")
        mlflow.log_metric(loss_name, value.item())


if __name__ == "__main__":
    main()
