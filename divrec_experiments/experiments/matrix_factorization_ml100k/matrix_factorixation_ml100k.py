from typing import Tuple

import torch
from torch.utils.data import DataLoader

from divrec.datasets import BPRSampling
from divrec.losses import LogSigmoidDifferenceLoss
from divrec.metrics import AUCScore, EntropyDiversityScore
from divrec.models import MatrixFactorization
from divrec_experiments.datasets import movie_lens_load, MovieLens100K


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


def main():
    path = "/Users/alexey.tsyplov/Projects/diversity-recommendations/divrec_experiments/data/ml-100k"
    dataset: MovieLens100K = movie_lens_load(path, train_size=0.7, test_size=0.2)

    train_dataset = BPRSampling(dataset.train, max_sampled=100)

    validation_dataset = BPRSampling(
        dataset.validation,
        user_item_interactions_frozen=dataset.train,
        max_sampled=200
    )

    test_dataset = BPRSampling(
        dataset.test,
        user_item_interactions_frozen=torch.concatenate((dataset.train, dataset.validation), dim=0).long(),
        max_sampled=-1,
    )

    train_loader = DataLoader(train_dataset, batch_size=1000)
    validation_loader = DataLoader(validation_dataset, batch_size=1000)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    model = MatrixFactorization(dataset.no_users, dataset.no_items, embedding_dim=300)
    model.to("cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = LogSigmoidDifferenceLoss()
    score_function = AUCScore()

    epochs = 10
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

    test_auc_score = evaluate_model(test_loader, model, score_function)
    print(f"Test AUC: {test_auc_score:.6f}")

    predictions = model.predict_top_k(10)
    entropy_diversity_score = EntropyDiversityScore()
    entropy_diversity_value = entropy_diversity_score(predictions).item()
    print(f"Diversity by entropy: {entropy_diversity_value:.6f}")


if __name__ == "__main__":
    main()
