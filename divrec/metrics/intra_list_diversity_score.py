from typing import Optional
from itertools import combinations
from divrec.utils import ScoreWithReduction
import torch


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
