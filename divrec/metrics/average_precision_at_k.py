import torch

from divrec.losses.base_losses import RecommendationsAwareLoss


def average_precision_at_k(
    interactions: torch.LongTensor, recommendations: torch.LongTensor
):
    """

    :param interactions: torch.LongTensor with two columns [user_id, item_id],
    asserts user_id = 0, 1, ..., no_users - 1
    :param recommendations: torch.LongTensor of size (no_users, k) with top K recommendations
    for each user
    :return: AveragePrecision@k, number of relevant items / k
    """
    no_users, k = recommendations.size()
    discount = torch.arange(1, k + 1)
    loss_values = torch.zeros(no_users)
    for user_id in range(no_users):
        positives = interactions[interactions[:, 0] == user_id, 1]
        relevance = torch.isin(recommendations[user_id], positives).float()
        loss_values[user_id] = torch.sum(torch.cumsum(relevance, 0) / discount)
    return loss_values / k


class AveragePrecisionAtKScore(RecommendationsAwareLoss):
    def recommendations_loss(
        self, interactions: torch.LongTensor, recommendations: torch.LongTensor
    ) -> torch.Tensor:
        return average_precision_at_k(interactions, recommendations)


class MeanAveragePrecisionAtKScore(AveragePrecisionAtKScore):
    def forward(
        self, interactions: torch.LongTensor, recommendations: torch.LongTensor
    ):
        return torch.mean(average_precision_at_k(interactions, recommendations))
