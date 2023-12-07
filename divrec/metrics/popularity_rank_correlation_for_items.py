import torch

from divrec.losses.base_losses import DatasetAwareLoss, RecommendationsAwareLoss


def rank(a: torch.Tensor, dim=-1, descending=False, stable=False):
    return torch.argsort(
        torch.argsort(a, dim=dim, descending=descending, stable=stable),
        dim=dim,
        stable=stable,
    )


def spearman_rank_correlation(
    a: torch.Tensor, b: torch.Tensor, evaluate_rank: bool = True
) -> torch.Tensor:
    assert a.size(0) == b.size(0)
    n = a.size(0)
    if evaluate_rank:
        d = rank(a) - rank(b)
        return 1 - 6 * torch.sum(d**2) / n / (n**2 - 1)
    a_std, a_mean = torch.std_mean(a)
    b_std, b_mean = torch.std_mean(b)
    return torch.mean((a - a_mean) * (b - b_mean)) / a_std / b_std


def avg_rank(recommendations: torch.LongTensor):
    ranks = {}
    for user in range(recommendations.size(0)):
        for index, item in enumerate(recommendations[user]):
            if item.item() in ranks:
                ranks[item.item()].append(index)
            else:
                ranks[item.item()] = [index]
    average_ranks = [[item, sum(r) / len(r)] for item, r in ranks.items()]
    items, average_ranks = zip(*average_ranks)
    items = torch.LongTensor(items)
    average_ranks = torch.FloatTensor(average_ranks)
    order = torch.argsort(items)
    return items[order], average_ranks[order]


class PRI(RecommendationsAwareLoss, DatasetAwareLoss):
    def __init__(self, *args, **kwargs):
        DatasetAwareLoss.__init__(self, *args, **kwargs)
        RecommendationsAwareLoss.__init__(self, *args, **kwargs)
        items, counts = torch.unique(
            self.dataset.interactions[:, 1], return_counts=True
        )
        self.popularity = torch.zeros(self.dataset.number_of_items, dtype=torch.float)
        self.popularity[items] = counts.float()
        self.popularity_rank = rank(self.popularity, descending=True).float()

    def forward(
        self, interactions: torch.LongTensor, recommendations: torch.LongTensor
    ):
        return self.recommendations_loss(interactions, recommendations)

    def recommendations_loss(
        self, interactions: torch.LongTensor, recommendations: torch.LongTensor
    ) -> torch.Tensor:
        items, ranks = avg_rank(recommendations)
        return spearman_rank_correlation(self.popularity_rank[items], ranks, evaluate_rank=False)
