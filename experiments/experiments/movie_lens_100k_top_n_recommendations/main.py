import os
from typing import Optional

import click
import mlflow
import torch
from divrec_experiments.datasets import movie_lens_load
from divrec_experiments.utils import (
    load_yaml,
    get_logger,
    create_if_not_exist,
    seed_everything,
)

from divrec.datasets import (
    PairWiseDataset,
    RankingDataset,
)
from divrec.metrics import (
    AUCScore,
    EntropyDiversityScore,
    MeanAveragePrecisionAtKScore,
    NDCGScore,
    PRI,
    PrecisionAtKScore,
    RecallAtKScore,
)
from divrec.models import RankingModel
from divrec.train import (
    pair_wise_score_loop,
    recommendations_score_loop,
)


class TopRecommendations(RankingModel):
    def __init__(self, no_items: int, interactions: torch.LongTensor):
        torch.nn.Module.__init__(self)
        self.no_items = no_items
        items, popularity = torch.unique(interactions[:, 1], return_counts=True)
        self.popularity = torch.zeros(no_items)
        self.popularity[items] = popularity.float()

    def forward(
        self,
        user_id: torch.LongTensor,
        item_id: torch.LongTensor,
        user_features: Optional[torch.Tensor],
        item_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self.popularity[item_id]


@click.command()
@click.argument("config_path")
def main(config_path: str) -> None:
    # --- instantiate config, logger and mlflow client ---
    config = load_yaml(config_path)
    workdir = config["workdir"] if "workdir" in config else os.path.abspath("workdir")
    create_if_not_exist(workdir)
    logger = get_logger(
        config["mlflow_experiment_name"],
        filepath=os.path.join(workdir, config["logfile"]),
    )
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["mlflow_experiment_name"])
    mlflow.log_artifact(config_path)
    seed_everything(config["seed"])

    # --- load and preprocess dataset ---
    dataset = movie_lens_load(config["data_directory"])
    logger.info("Successfully load dataset")
    mlflow.log_param("number_of_users", dataset.train.number_of_users)
    mlflow.log_param("number_of_items", dataset.train.number_of_items)
    mlflow.log_param(
        "number_of_interactions",
        dataset.train.number_of_interactions + dataset.test.number_of_interactions,
    )
    mlflow.log_param("train_interactions_count", dataset.train.number_of_interactions)
    mlflow.log_param("test_interactions_count", dataset.test.number_of_interactions)

    # --- model instantiating ---
    model = TopRecommendations(
        dataset.train.number_of_items, dataset.train.interactions
    )
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
    mlflow.log_metric("test_auc_score", scores[0].item())

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
