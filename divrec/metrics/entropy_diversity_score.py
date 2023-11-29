import math
import torch


class EntropyDiversityScore(torch.nn.Module):
    def forward(self, recommendations: torch.Tensor):
        _, counts = torch.unique(recommendations, return_counts=True)
        probabilities = counts / counts.sum()
        actual = -torch.sum(probabilities * torch.log(probabilities))
        ideal = math.log(counts.size(0))
        return actual / ideal
