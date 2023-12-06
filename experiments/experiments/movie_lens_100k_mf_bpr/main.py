from typing import Tuple

import os
import click
import mlflow
import torch

from divrec.datasets import (
    UserItemInteractionsDataset,
    PointWiseDataset,
    PairWiseDataset,
    RankingDataset,
)
from divrec.losses import MSELoss
from divrec.metrics import (
    AUCScore,
    EntropyDiversityScore,
    MeanAveragePrecisionAtKScore,
    NDCGScore,
    PRI,
    PrecisionAtKScore,
    RecallAtKScore,
)
from divrec.models import MatrixFactorization
from divrec.train import (
    point_wise_train_loop,
    pair_wise_score_loop,
    recommendations_score_loop,
)
from divrec_experiments.datasets import movie_lens_load
from divrec_experiments.utils import load_yaml, get_logger, create_if_not_exist


def train_validation_split(
    dataset: UserItemInteractionsDataset, validation_size: int
) -> Tuple[UserItemInteractionsDataset, UserItemInteractionsDataset]:
    """

    :param dataset: train data
    :param validation_size: number of items per user used for validation
    :return: tuple of train and validation dataset
    """
    train_interactions = list()
    train_interactions_scores = list()

    validation_interactions = list()
    validation_interactions_scores = list()

    for user_id in range(dataset.number_of_users):
        binary_mask = dataset.interactions[:, 0] == user_id
        user_interactions = dataset.interactions[binary_mask]
        user_interactions_scores = dataset.interaction_scores[binary_mask]

        train_interactions.append(user_interactions[:-validation_size])
        train_interactions_scores.append(user_interactions_scores[:-validation_size])

        validation_interactions.append(user_interactions[-validation_size:])
        validation_interactions_scores.append(
            user_interactions_scores[-validation_size:]
        )

    train = UserItemInteractionsDataset(
        interactions=torch.concatenate(train_interactions).long(),
        interaction_scores=torch.concatenate(train_interactions_scores),
        item_features=dataset.item_features,
        user_features=dataset.user_features,
        number_of_items=dataset.number_of_items,
        number_of_users=dataset.number_of_users,
    )

    validation = UserItemInteractionsDataset(
        interactions=torch.concatenate(validation_interactions).long(),
        interaction_scores=torch.concatenate(validation_interactions_scores),
        item_features=dataset.item_features,
        user_features=dataset.user_features,
        number_of_items=dataset.number_of_items,
        number_of_users=dataset.number_of_users,
    )

    return train, validation


@click.command()
@click.argument("config_path")
def main(config_path: str) -> None:
    # --- instantiate config, logger and mlflow client ---
    config = load_yaml(config_path)
    workdir = config["workdir"] if "workdir" in config else os.path.abspath("workdir")
    create_if_not_exist(workdir)
    logger = get_logger(config["mlflow_experiment_name"], filepath=os.path.join(workdir, config["logfile"]))
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["mlflow_experiment_name"])
    mlflow.log_artifact(config_path)

    # --- load and preprocess dataset ---
    dataset = movie_lens_load(config["data_directory"])
    logger.info("Successfully load dataset")
    mlflow.log_param("number_of_users", dataset.train.number_of_users)
    mlflow.log_param("number_of_items", dataset.train.number_of_items)
    mlflow.log_param(
        "number_of_interactions",
        dataset.train.number_of_interactions + dataset.test.number_of_interactions,
    )

    # --- prepare train validation split ---
    train, validation = train_validation_split(
        dataset.train, config["items_per_user_for_validation"]
    )
    test = dataset.test
    logger.info("Successfully split data for train, validation and test")
    mlflow.log_param("train_interactions_count", train.number_of_interactions)
    mlflow.log_param("validation_interactions_count", validation.number_of_interactions)
    mlflow.log_param("test_interactions_count", test.number_of_interactions)

    # --- model and optimizer instantiating ---
    model = MatrixFactorization(
        train.number_of_users,
        train.number_of_items,
        embedding_dim=config["embedding_dim"],
    )
    model.to(config["device"])

    optimizer = torch.optim.Adam(model.parameters(), **config["optimizer"])
    loss_functions = MSELoss()

    # --- train stage ---
    train_dataset = PointWiseDataset(train)
    validation_dataset = PairWiseDataset(
        validation, frozen=train, max_sampled=config["validation_max_sampled"]
    )
    epochs = config["epochs"]
    for epoch in range(config["epochs"]):
        mse, _ = point_wise_train_loop(
            train_dataset, model, loss_functions, optimizer, **config["train_loader"]
        )
        logger.info(f"Epoch [{epoch + 1}/{epochs}] train MSE: {mse:.6f}")
        mlflow.log_metric("mse_loss", mse, step=epoch)

        scores = pair_wise_score_loop(
            validation_dataset, model, [AUCScore()], **config["validation_loader"]
        )
        logger.info(f"Epoch [{epoch + 1}/{epochs}] validation AUC: {scores[0]:.6f}")
        mlflow.log_metric("auc_score", scores[0], step=epoch)

    logger.info("Successfully finished model train")

    # --- model saving ---
    torch.save(model.state_dict(), os.path.join(workdir, config["model_path"]))
    logger.info("Successfully finished model saving")

    # --- test stage ---
    test_ranking_dataset = RankingDataset(test, frozen=dataset.train)
    test_pairwise_dataset = PairWiseDataset(
        test, frozen=dataset.train, max_sampled=config["test_max_sampled"]
    )

    scores = pair_wise_score_loop(
        test_pairwise_dataset, model, [AUCScore()], **config["test_pairwise_loader"],
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
        test_ranking_dataset, model, losses, number_of_recommendations=config["k"],
    )

    for loss, value in zip(losses, loss_values):
        loss_name = f"{loss.name}_at_{config['k']}"
        logger.info(f"test {loss_name}: {value.item()}")
        mlflow.log_metric(loss_name, value.item())


if __name__ == "__main__":
    main()
