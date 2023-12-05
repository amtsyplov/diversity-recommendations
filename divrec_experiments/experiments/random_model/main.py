import os
from typing import Any, Dict

import yaml
import click

from divrec.utils import get_logger
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


def load_config(path: str) -> Dict[str, Any]:
    with open(os.path.abspath(path), mode="r") as file:
        return yaml.safe_load(file)


@click.command()
@click.argument("config_path")
def main(config_path: str) -> None:
    config = load_config(config_path)

    logger = get_logger("random_model_experiment", filepath=config.get("logfile"))

    datasets = movie_lens_load(config["data_directory"])
    logger.info("Successfully load data")

    model = RandomModel(datasets.test.number_of_users, datasets.test.number_of_items)
    model.to(config["device"])

    pair_wise_test_dataset = PairWiseDataset(datasets.test, frozen=datasets.train, max_sampled=20)

    auc_score = pair_wise_score_loop(
        pair_wise_test_dataset,
        model,
        [AUCScore()],
        batch_size=200,
    )
    logger.info(f"Successfully evaluate AUC on test: {auc_score[0].item()}")

    losses = [
        EntropyDiversityScore(dataset=datasets.train),
        MeanAveragePrecisionAtKScore(),
        NDCGScore(),
        PRI(dataset=datasets.train),
        PrecisionAtKScore(),
        RecallAtKScore()
    ]
    ranking_test_dataset = RankingDataset(datasets.test, frozen=datasets.train)
    loss_values = recommendations_score_loop(
        ranking_test_dataset,
        model,
        losses,
        number_of_recommendations=config["k"]
    )
    for loss, value in zip(losses, loss_values):
        logger.info(f"Successfully evaluate {loss.__class__.__name__}@{config['k']} on test: {value.item()}")


if __name__ == '__main__':
    main()
