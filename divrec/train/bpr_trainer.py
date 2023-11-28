from typing import Optional

import torch
from torch.utils.data import DataLoader

from divrec.datasets import BPRSampling
from divrec.losses import LogSigmoidDifferenceLoss
from divrec.train import Trainer


class BPRTrainer(Trainer):
    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            score_function: torch.nn.Module,
            train: BPRSampling,
            validation: Optional[BPRSampling] = None,
            train_batch_size: int = 128,
            validation_batch_size: int = 128,
            epochs: int = 10,
            logfile: Optional[str] = None,
    ):
        self.train = train
        self.validation = validation
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        Trainer.__init__(
            self,
            model,
            optimizer,
            LogSigmoidDifferenceLoss(),
            score_function,
            DataLoader(train, batch_size=train_batch_size),
            validation_loader=DataLoader(validation, batch_size=train_batch_size),
            epochs=epochs,
            logfile=logfile,
        )

    def fit_partial(self, validation_mode: bool = False):
        loader = self.validation_loader if validation_mode else self.train_loader
        loss_value = 0.0
        score_value = 0.0
        batch_count = 0

        if validation_mode:
            self.model.eval()
        else:
            self.model.train()

        for user, positive, negative in loader:
            positive_predictions = self.model(user, positive)
            negative_predictions = self.model(user, negative)

            loss = self.loss_function(positive_predictions, negative_predictions)
            score = self.score_function(positive_predictions, negative_predictions)

            if not validation_mode:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            loss_value += loss.item()
            score_value += score.item()
            batch_count += 1

        loss_value /= batch_count
        score_value /= batch_count

        return loss_value, score_value
