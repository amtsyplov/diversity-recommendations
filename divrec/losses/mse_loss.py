import torch

from divrec.losses.base_losses import PointWiseLoss


class MSELoss(PointWiseLoss):
    def point_wise(
        self, true_relevance: torch.Tensor, predicted_relevance: torch.Tensor
    ) -> torch.Tensor:
        return (true_relevance - predicted_relevance) ** 2
