import torch

from divrec.losses.base_losses import PairWiseLoss


class MSEDifferenceLoss(PairWiseLoss):
    def pair_wise(
        self, positives: torch.Tensor, negatives: torch.Tensor
    ) -> torch.Tensor:
        return (1 - (positives - negatives)) ** 2
