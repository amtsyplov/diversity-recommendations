import torch

from divrec.utils import ScoreWithReduction


def recall_at_k(interactions: torch.LongTensor, recommendations: torch.LongTensor):
    """

    :param interactions: torch.LongTensor with two columns [user_id, item_id],
    asserts user_id = 0, 1, ..., no_users - 1
    :param recommendations: torch.LongTensor of size (no_users, k) with top K recommendations
    for each user
    :return: Recall@k, number of relevant items / k
    """
    no_users = recommendations.size(0)
    loss_values = torch.zeros(no_users)
    for user_id in range(no_users):
        positives = interactions[interactions[:, 0] == user_id, 1]
        loss_values[user_id] = torch.isin(recommendations[user_id], positives).sum().item() / positives.size(0)
    return loss_values


class RecallAtKScore(ScoreWithReduction):
    def forward(self, interactions: torch.LongTensor, recommendations: torch.LongTensor):
        return self.reduce_loss_values(recall_at_k(interactions, recommendations))
