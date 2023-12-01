import torch

from divrec.losses.base_losses import RecommendationsAwareLoss


def normalized_discounted_cumulative_gain(
    interactions: torch.LongTensor, recommendations: torch.LongTensor
):
    """
    :param interactions: torch.LongTensor with two columns [user_id, item_id],
    asserts user_id = 0, 1, ..., no_users - 1
    :param recommendations: torch.LongTensor of size (no_users, k) with top K recommendations
    for each user
    :return: NDCG@k
    """
    no_users, k = recommendations.size()
    discount = torch.log2(torch.arange(2, k + 2))
    loss_values = torch.zeros(no_users)
    for user_id in range(no_users):
        positives = interactions[interactions[:, 0] == user_id, 1]
        cumulative_gain = torch.isin(recommendations[user_id], positives).float()
        loss_values[user_id] = torch.sum(cumulative_gain / discount)
    return loss_values / torch.sum(1 / discount)


class NDCGScore(RecommendationsAwareLoss):
    def recommendations_loss(
        self, interactions: torch.LongTensor, recommendations: torch.LongTensor
    ) -> torch.Tensor:
        return normalized_discounted_cumulative_gain(interactions, recommendations)
