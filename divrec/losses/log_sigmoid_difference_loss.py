import torch

from divrec.losses.base_losses import PairWiseLoss


class LogSigmoidDifferenceLoss(PairWiseLoss):
    def __init__(self, *args, **kwargs):
        PairWiseLoss.__init__(self, *args, *kwargs)
        self.log_sigmoid = torch.nn.LogSigmoid()

    def pair_wise(
        self, positives: torch.Tensor, negatives: torch.Tensor
    ) -> torch.Tensor:
        return -self.log_sigmoid(positives - negatives)
