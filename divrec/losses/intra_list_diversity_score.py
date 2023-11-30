from itertools import combinations
import torch

from divrec.losses.base_losses import RecommendationsAwareLoss, DatasetAwareLoss


class IntraListDiversityScore(RecommendationsAwareLoss):
    """
    Diversity metric written in
    
    Incorporating Diversity in a Learning to Rank Recommender System
    by Jacek Wasilewski and Neil Hurley
    """
    def __init__(self, *args, distance_matrix: torch.Tensor, **kwargs):
        RecommendationsAwareLoss.__init__(self, *args, **kwargs)
        self.distance_matrix = distance_matrix

    def recommendations_loss(self, interactions: torch.LongTensor, recommendations: torch.LongTensor) -> torch.Tensor:
        no_user_recommendations = recommendations.size(1)

        loss_values = torch.Tensor(
            [self.user_ild(user_recommendations, self.distance_matrix) for user_recommendations in recommendations]
        )

        loss_values /= no_user_recommendations * (no_user_recommendations - 1)

        return self.reduce_loss_values(loss_values)

    @staticmethod
    def user_ild(user_recommendations: torch.Tensor, distance_matrix: torch.Tensor) -> float:
        return sum(distance_matrix[i, j] for i, j in combinations(user_recommendations, 2))


class IntraListBinaryUnfairnessScore(IntraListDiversityScore, DatasetAwareLoss):
    """

    Controlling Popularity Bias in Learning to Rank Recommendation
    by Himan Abdollahpouri, Robin Burke and others
    """
    def __init__(self, *args, items_partition_feature: str = "partition", **kwargs):
        DatasetAwareLoss.__init__(self, *args, **kwargs)
        self.items_partition_feature = items_partition_feature

        IntraListDiversityScore.__init__(
            self,
            *args,
            distance_matrix=self.get_distance_matrix(),
            **kwargs
        )

    def get_distance_matrix(self):
        assert self.items_partition_feature in self.dataset.item_features
        item_labels = self.dataset.item_features[self.items_partition_feature]
        return (item_labels.unsqueeze(0) == item_labels.unsqueeze(1)).int()
