from typing import Optional

import torch

from divrec.datasets import UserItemInteractionsDataset
from divrec.utils import to_camel_case


class DatasetAwareLoss:
    def __init__(
        self, *args, dataset: Optional[UserItemInteractionsDataset] = None, **kwargs
    ):
        self.dataset = dataset


class ScoreWithReduction:
    def __init__(self, *args, reduce: bool = True, reduction: str = "mean", **kwargs):
        assert reduction in ["none", "mean", "sum"]
        self.reduce = reduce and reduction != "none"
        self.reduction = reduction

    def reduce_loss_values(self, loss_values: torch.Tensor):
        if not self.reduce:
            return loss_values
        elif self.reduction == "mean":
            return torch.sum(loss_values, dim=0) / loss_values.size(0)
        return torch.sum(loss_values, dim=0)

    @property
    def name(self):
        return to_camel_case(type(self).__name__)


class PointWiseLoss(torch.nn.Module, ScoreWithReduction):
    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        ScoreWithReduction.__init__(self, *args, **kwargs)

    def forward(self, true_relevance: torch.Tensor, predicted_relevance: torch.Tensor):
        return self.reduce_loss_values(
            self.point_wise(true_relevance, predicted_relevance)
        )

    def point_wise(
        self, true_relevance: torch.Tensor, predicted_relevance: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError(
            f'Score [{type(self).__name__}] is missing the required "point_wise" function'
        )


class PairWiseLoss(torch.nn.Module, ScoreWithReduction):
    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        ScoreWithReduction.__init__(self, *args, **kwargs)

    def forward(self, positives: torch.Tensor, negatives: torch.Tensor):
        return self.reduce_loss_values(self.pair_wise(positives, negatives))

    def pair_wise(
        self, positives: torch.Tensor, negatives: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError(
            f'Score [{type(self).__name__}] is missing the required "pair_wise" function'
        )


class RecommendationsAwareLoss(torch.nn.Module, ScoreWithReduction):
    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        ScoreWithReduction.__init__(self, *args, **kwargs)

    def forward(
        self, interactions: torch.LongTensor, recommendations: torch.LongTensor
    ):
        return self.reduce_loss_values(
            self.recommendations_loss(interactions, recommendations)
        )

    def recommendations_loss(
        self, interactions: torch.LongTensor, recommendations: torch.LongTensor
    ) -> torch.Tensor:
        raise NotImplementedError(
            f'Score [{type(self).__name__}] is missing the required "recommendations_loss" function'
        )
