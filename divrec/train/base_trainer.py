from abc import ABCMeta, abstractmethod
from typing import Optional

import torch
from torch.utils.data import DataLoader

from divrec.utils import get_logger


class Trainer(metaclass=ABCMeta):
    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            loss_function: torch.nn.Module,
            score_function: torch.nn.Module,
            train_loader: DataLoader,
            validation_loader: Optional[DataLoader] = None,
            epochs: int = 10,
            logfile: Optional[str] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.score_function = score_function
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.epochs = epochs
        self.logger = get_logger(self.__class__.__name__, filepath=logfile)

    def fit(self):
        train_losses = []
        train_scores = []
        validation_losses = []
        validation_scores = []
        for epoch in range(self.epochs):
            batch_avg_loss, batch_avg_score = self.fit_partial()
            train_losses.append(batch_avg_loss)
            train_scores.append(batch_avg_score)
            self.logger.info("Train " + self.epoch_message(epoch, batch_avg_loss, batch_avg_score))

            if self.validation_loader is not None:
                batch_avg_loss, batch_avg_score = self.fit_partial(validation_mode=True)
                validation_losses.append(batch_avg_loss)
                validation_scores.append(batch_avg_score)
                self.logger.info("Validation " + self.epoch_message(epoch, batch_avg_loss, batch_avg_score))

        return train_losses, train_scores, validation_losses, validation_scores

    def print_loss(self, loss) -> str:
        return f"{self.loss_function.__class__.__name__}: {loss:.6f}"

    def print_score(self, score) -> str:
        return f"{self.score_function.__class__.__name__}: {score:.6f}"

    def print_epoch(self, epoch) -> str:
        if self.epochs >= 1000:
            return f"Epoch [{epoch:4d}/{self.epochs:4d}]"
        elif self.epochs >= 100:
            return f"Epoch [{epoch:3d}/{self.epochs:3d}]"
        elif self.epochs >= 10:
            return f"Epoch [{epoch:2d}/{self.epochs:2d}]"
        return f"Epoch [{epoch}/{self.epochs}]"

    def epoch_message(self, epoch, loss, score):
        return f"{self.print_epoch(epoch)} {self.print_loss(loss)} {self.print_score(score)}"

    @abstractmethod
    def fit_partial(self, validation_mode: bool = False):
        raise NotImplemented
