from typing import Optional
from itertools import combinations

import torch


class ScoreWithReduction(torch.nn.Module):
    def __init__(self, reduce: bool = True, reduction: str = "mean"):
        assert reduction in ["node", "mean", "sum"]
        torch.nn.Module.__init__(self)
        self.reduce = reduce and reduction != "none"
        self.reduction = reduction

    def reduce_loss_values(self, loss_values: torch.Tensor):
        if not self.reduce:
            return loss_values
        elif self.reduction == "mean":
            return torch.mean(loss_values)
        return torch.sum(loss_values)


class AUCScore(ScoreWithReduction):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss_values = (x >= y).long()
        return self.reduce_loss_values(loss_values)


class IntraListDiversityScore(ScoreWithReduction):
    """
    Diversity metric written in
    
    Incorporating Diversity in a Learning to Rank Recommender System
    by JacekWasilewski and Neil Hurley
    """
    def __init__(self, *args, distance_matrix: Optional[torch.Tensor] = None, **kwargs):
        ScoreWithReduction.__init__(self, *args, **kwargs)
        self.distance_matrix = distance_matrix

    def forward(self, recommendations: torch.Tensor, distance_matrix: Optional[torch.Tensor] = None):
        distances = self.distance_matrix if distance_matrix is None else distance_matrix
        if distances is None:
            raise ValueError("Distance matrix is not specified")

        no_user_recommendations = recommendations.size(1)

        loss_values = torch.Tensor(
            [self.user_ild(user_recommendations, distances) for user_recommendations in recommendations]
        )

        loss_values /= no_user_recommendations * (no_user_recommendations - 1)

        return self.reduce_loss_values(loss_values)

    @staticmethod
    def user_ild(user_recommendations: torch.Tensor, distance_matrix: torch.Tensor) -> float:
        return sum(distance_matrix[i, j] for i, j in combinations(user_recommendations, 2))


class EntropyDiversityScore(torch.nn.Module):
    def forward(self, recommendations: torch.Tensor):
        _, counts = torch.unique(recommendations, return_counts=True)
        probabilities = counts / counts.sum()
        return -torch.sum(probabilities * torch.log(probabilities))
