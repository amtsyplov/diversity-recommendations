import torch

from divrec.utils import ScoreWithReduction


class AUCScore(ScoreWithReduction):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss_values = (x >= y).long()
        return self.reduce_loss_values(loss_values)
