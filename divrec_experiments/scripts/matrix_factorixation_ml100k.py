from divrec.datasets import BPRSampling
from divrec.models import MatrixFactorization
from divrec.train import bpr_train_loop
from divrec.losses import (
    AUCScore,
    IntraListDiversityScore,
    EntropyDiversityScore,
    LogSigmoidDifferenceLoss
)

from divrec_experiments.datasets import movie_lens_load, MovieLens100K

import os
import torch
from torch.utils.data import DataLoader


def main():
    path = "/Users/alexey.tsyplov/Projects/diversity-recommendations/divrec_experiments/data/ml-100k"
    dataset: MovieLens100K = movie_lens_load(path)

    train_dataset = BPRSampling(dataset.train)
    validation_dataset = BPRSampling(dataset.validation)
    test_dataset = BPRSampling(dataset.test)

    train_loader = DataLoader(train_dataset, batch_size=1000)
    validation_loader = DataLoader(validation_dataset, batch_size=1000)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    model = MatrixFactorization(dataset.no_users, dataset.no_items, embedding_dim=300)
    model.to("cpu")

    optimizer = torch.optim.Adam(model, lr=0.001)
    loss_function = LogSigmoidDifferenceLoss()
    score_function = AUCScore()

    epochs = 100
    for epoch in range(epochs):
        model.train()
        batch_avg_loss, batch_avg_score = bpr_train_loop(
            train_loader, model, optimizer, loss_function, score_function=score_function
        )
        print(f"Train epoch [{epoch}/{epochs}], BPR {batch_avg_loss:.6f}, AUC {batch_avg_score:.6f}")

        model.eval()
        batch_avg_loss, batch_avg_score = bpr_train_loop(
            validation_loader, model, optimizer, loss_function, score_function=score_function
        )
        print(f"Validation BPR {batch_avg_loss:.6f}, AUC {batch_avg_score:.6f}")

