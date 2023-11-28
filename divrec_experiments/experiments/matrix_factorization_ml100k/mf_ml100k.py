from typing import Tuple

import torch
from torch.utils.data import DataLoader

from divrec.datasets import BPRSampling
from divrec.losses import LogSigmoidDifferenceLoss
from divrec.metrics import AUCScore, EntropyDiversityScore
from divrec.models import MatrixFactorization
from divrec_experiments.datasets import movie_lens_load, MovieLens100K
from divrec_experiments.pipeline import Container, Pipeline, stage


@stage(configuration={
    "path": "data/ml-100k",
    "train_size": 0.7,
    "test_size": 0.2,
})
def load_dataset(config, arg):
    data = movie_lens_load(config["path"], config["train_size"], config["test_size"])
    return Container(elements={"data": data})


def train_loop(
        loader: DataLoader,
        model: MatrixFactorization,
        optimizer: torch.optim.Optimizer,
        loss_function: torch.nn.Module,
        score_function: torch.nn.Module,
        validation_mode: bool = False
) -> Tuple[float, float]:
    loss_value = 0.0
    score_value = 0.0
    batch_count = 0

    if not validation_mode:
        model.train()
    else:
        model.eval()

    for user, positive, negative in loader:
        positive_predictions = model(user, positive)
        negative_predictions = model(user, negative)

        loss = loss_function(positive_predictions, negative_predictions)
        score = score_function(positive_predictions, negative_predictions)

        if not validation_mode:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        loss_value += loss.item()
        score_value += score.item()
        batch_count += 1

    loss_value /= batch_count
    score_value /= batch_count

    return loss_value, score_value


def evaluate_model(
        loader: DataLoader,
        model: MatrixFactorization,
        score_function: torch.nn.Module
) -> float:
    score_value = 0.0
    batch_count = 0

    model.eval()
    for user, positive, negative in loader:
        positive_predictions = model(user, positive)
        negative_predictions = model(user, negative)

        score = score_function(positive_predictions, negative_predictions)
        score_value += score.item()
        batch_count += 1

    return score_value / batch_count


@stage(configuration={
    "train_max_sampled": 100,
    "train_batch_size": 100,
    "validation_max_sampled": 100,
    "validation_batch_size": 100,
    "embedding_dim": 300,
    "lr": 0.001,
    "epochs": 10,
    "model_path": "models"
})
def train_model(config, arg):
    dataset: MovieLens100K = arg["data"]
    train_dataset = BPRSampling(dataset.train, max_sampled=config["train_max_sampled"])

    validation_dataset = BPRSampling(
        dataset.validation,
        user_item_interactions_frozen=dataset.train,
        max_sampled=config["validation_max_sampled"]
    )

    train_loader = DataLoader(train_dataset, batch_size=config["train_batch_size"])
    validation_loader = DataLoader(validation_dataset, batch_size=config["validation_batch_size"])

    model = MatrixFactorization(dataset.no_users, dataset.no_items, embedding_dim=config["embedding_dim"])
    model.to("cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_function = LogSigmoidDifferenceLoss()
    score_function = AUCScore()

    epochs = config["epochs"]
    for epoch in range(epochs):
        batch_avg_loss, batch_avg_score = train_loop(
            train_loader,
            model,
            optimizer,
            loss_function,
            score_function,
        )
        print(f"Train epoch [{epoch}/{epochs}], BPR {batch_avg_loss:.6f}, AUC {batch_avg_score:.6f}")

        batch_avg_loss, batch_avg_score = train_loop(
            validation_loader,
            model,
            optimizer,
            loss_function,
            score_function,
            validation_mode=True,
        )
        print(f"Validation BPR {batch_avg_loss:.6f}, AUC {batch_avg_score:.6f}")
    print("Train finished")

    torch.save(model.state_dict(), config["model_path"])
    return Container(elements={"data": dataset})


@stage(configuration={
    "model_path": "models",
    "embedding_dim": 300,
    "max_sampled": -1,
})
def test_model(config, arg):
    dataset: MovieLens100K = arg["data"]

    model = MatrixFactorization(dataset.no_users, dataset.no_items, embedding_dim=config["embedding_dim"])
    model.to("cpu")

    test_dataset = BPRSampling(
        dataset.test,
        user_item_interactions_frozen=torch.concatenate((dataset.train, dataset.validation), dim=0).long(),
        max_sampled=config["max_sampled"],
    )

    test_loader = DataLoader(test_dataset, batch_size=1000)
    score_function = AUCScore()

    model.load_state_dict(torch.load(config["model_path"]))

    test_auc_score = evaluate_model(test_loader, model, score_function)
    print(f"Test AUC: {test_auc_score:.6f}")

    predictions = model.predict_top_k(10)
    entropy_diversity_score = EntropyDiversityScore()
    entropy_diversity_value = entropy_diversity_score(predictions).item()
    print(f"Diversity by entropy: {entropy_diversity_value:.6f}")


pipeline = Pipeline([
    load_dataset,
    train_model,
    test_model
])


if __name__ == "__main__":
    pipeline.run()
