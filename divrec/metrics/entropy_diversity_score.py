import math
import torch

from divrec.losses.base_losses import RecommendationsAwareLoss


class EntropyDiversityScore(RecommendationsAwareLoss):
    def forward(
        self, interactions: torch.LongTensor, recommendations: torch.LongTensor
    ):
        _, counts = torch.unique(recommendations, return_counts=True)
        probabilities = counts / counts.sum()
        actual = -torch.sum(probabilities * torch.log(probabilities))
        ideal = math.log(counts.size(0))
        return actual / ideal
