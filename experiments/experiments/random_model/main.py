import os

import click
import mlflow
import torch
from divrec_experiments.datasets import movie_lens_load
from divrec_experiments.utils import create_if_not_exist, get_logger, seed_everything, load_yaml

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
from divrec.models import RandomModel
from divrec.train.utils import pair_wise_score_loop, recommendations_score_loop


@click.command()
@click.argument("config_path")
def main(config_path: str) -> None:
    # --- instantiate config, logger and mlflow client ---
    config = load_yaml(config_path)
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

    # --- load and preprocess dataset ---
    dataset = movie_lens_load(config["data_directory"])
    logger.info("Successfully load data")
    mlflow.log_param("number_of_users", dataset.train.number_of_users)
    mlflow.log_param("number_of_items", dataset.train.number_of_items)
    mlflow.log_param(
        "number_of_interactions",
        dataset.train.number_of_interactions + dataset.test.number_of_interactions,
    )
    mlflow.log_param("train_interactions_count", dataset.train.number_of_interactions)
    mlflow.log_param("test_interactions_count", dataset.test.number_of_interactions)

    # --- model instantiating ---
    model = RandomModel(dataset.test.number_of_users, dataset.test.number_of_items)
    model.to(config["device"])

    # --- model saving ---
    torch.save(model.state_dict(), os.path.join(workdir, config["model_path"]))
    logger.info("Successfully finished model saving")

    # --- test stage ---
    test_ranking_dataset = RankingDataset(dataset.test, frozen=dataset.train)
    test_pairwise_dataset = PairWiseDataset(
        dataset.test, frozen=dataset.train, max_sampled=config["test_max_sampled"]
    )

    scores = pair_wise_score_loop(
        test_pairwise_dataset,
        model,
        [AUCScore()],
        **config["test_pairwise_loader"],
    )
    logger.info(f"test AUC: {scores[0].item()}")
    mlflow.log_metric("auc_score", scores[0].item())

    losses = [
        EntropyDiversityScore(dataset=dataset.train),
        MeanAveragePrecisionAtKScore(),
        NDCGScore(),
        PRI(dataset=dataset.train),
        PrecisionAtKScore(),
        RecallAtKScore(),
    ]

    loss_values = recommendations_score_loop(
        test_ranking_dataset,
        model,
        losses,
        number_of_recommendations=config["k"],
    )

    for loss, value in zip(losses, loss_values):
        loss_name = f"{loss.name}_at_{config['k']}"
        logger.info(f"test {loss_name}: {value.item()}")
        mlflow.log_metric(loss_name, value.item())


if __name__ == "__main__":
    main()
