import math
import torch

from divrec.losses.base_losses import DatasetAwareLoss, RecommendationsAwareLoss


class EntropyDiversityScore(RecommendationsAwareLoss, DatasetAwareLoss):
    def __init__(self, *args, **kwargs):
        DatasetAwareLoss.__init__(self, *args, **kwargs)
        RecommendationsAwareLoss.__init__(self, *args, **kwargs)
        self.max_entropy = math.log(self.dataset.number_of_items)

    def forward(
        self, interactions: torch.LongTensor, recommendations: torch.LongTensor
    ):
        return self.recommendations_loss(interactions, recommendations)

    def recommendations_loss(
        self, interactions: torch.LongTensor, recommendations: torch.LongTensor
    ) -> torch.Tensor:
        _, counts = torch.unique(recommendations, return_counts=True)
        probabilities = counts / counts.sum()
        actual_entropy = -torch.sum(probabilities * torch.log(probabilities))
        min_entropy = math.log(recommendations.size(1))
        return (actual_entropy - min_entropy) / (self.max_entropy - min_entropy)
