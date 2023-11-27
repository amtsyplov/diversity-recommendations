import torch


class EntropyDiversityScore(torch.nn.Module):
    def forward(self, recommendations: torch.Tensor):
        _, counts = torch.unique(recommendations, return_counts=True)
        probabilities = counts / counts.sum()
        return -torch.sum(probabilities * torch.log(probabilities))
