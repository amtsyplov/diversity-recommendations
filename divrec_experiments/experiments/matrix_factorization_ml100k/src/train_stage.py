import torch

from divrec.datasets import BPRSampling
from divrec.metrics import AUCScore
from divrec.models import MatrixFactorization
from divrec.train import BPRTrainer
from divrec_experiments.datasets import MovieLens100K
from divrec_pipelines.pipeline import Container, stage


@stage(configuration={
    "train_max_sampled": 100,
    "train_batch_size": 100,
    "validation_max_sampled": 100,
    "validation_batch_size": 100,
    "embedding_dim": 300,
    "lr": 0.001,
    "epochs": 10,
    "model_path": "workdir",
    "logfile": "logfile.log"
})
def train_model(config, arg):
    dataset: MovieLens100K = arg["data"]
    train_dataset = BPRSampling(dataset.train, max_sampled=config["train_max_sampled"])

    validation_dataset = BPRSampling(
        dataset.validation,
        user_item_interactions_frozen=dataset.train,
        max_sampled=config["validation_max_sampled"]
    )

    model = MatrixFactorization(
        dataset.no_users,
        dataset.no_items,
        embedding_dim=config["embedding_dim"]
    )
    model.to("cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    score_function = AUCScore()

    trainer = BPRTrainer(
        model,
        optimizer,
        score_function,
        train_dataset,
        validation_dataset,
        train_batch_size=config["train_batch_size"],
        validation_batch_size=config["validation_batch_size"],
        epochs=config["epochs"],
        logfile=config["logfile"],
    )

    trainer.fit()

    torch.save(model.state_dict(), config["model_path"])
    return Container(elements={"data": dataset})
