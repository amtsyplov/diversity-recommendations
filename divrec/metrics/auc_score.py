import torch

from divrec.losses.base_losses import PairWiseLoss


class AUCScore(PairWiseLoss):
    def pair_wise(
        self, positives: torch.Tensor, negatives: torch.Tensor
    ) -> torch.Tensor:
        return (positives >= negatives).int()
