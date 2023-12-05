from typing import List, Optional
from dataclasses import dataclass, field

import torch


@dataclass
class Features:
    """
    Abstraction for table with named columns
    """

    features: torch.Tensor = None
    feature_names: List[str] = field(default_factory=list)

    number_of_features: int = 0

    def __post_init__(self):
        self.number_of_features = max(self.number_of_features, self.features.size(1))
        assert len(self.feature_names) == self.number_of_features

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, item):
        """Gives column by name"""
        return self.features[:, self.feature_names.index(item)]

    def __contains__(self, item):
        return item in self.feature_names


@dataclass
class UserItemInteractionsDataset:
    """
    Standard class for recommendation systems training and inference.
    Includes user-item interactions, its' scores, users' and items'
    features. All data parts are optional because depends on real
    datasets.
    """

    interactions: Optional[torch.LongTensor] = None
    interaction_scores: Optional[torch.Tensor] = None
    user_features: Optional[Features] = None
    item_features: Optional[Features] = None

    number_of_interactions: int = 0
    number_of_users: int = 0
    number_of_items: int = 0

    def __post_init__(self):
        if self.interactions is not None:
            self.number_of_interactions, no_columns = self.interactions.size()
            assert no_columns == 2

            if self.interaction_scores is None:
                self.interaction_scores = torch.ones(self.number_of_interactions)
            assert self.interaction_scores.size(0) == self.number_of_interactions

            users = torch.unique(self.interactions[:, 0])
            items = torch.unique(self.interactions[:, 1])

            self.number_of_users = max(self.number_of_users, users.max() + 1)
            self.number_of_items = max(self.number_of_items, items.max() + 1)

            assert torch.all(torch.isin(users, torch.arange(self.number_of_users))).item()
            assert torch.all(torch.isin(items, torch.arange(self.number_of_items))).item()

        if self.interaction_scores is not None:
            assert self.interactions is not None
            assert self.interaction_scores.size(0) == self.number_of_interactions

        if self.user_features is not None:
            if self.number_of_users == 0:
                self.number_of_users = len(self.user_features)
            assert self.number_of_users == len(self.user_features)

        if self.item_features is not None:
            if self.number_of_items == 0:
                self.number_of_items = len(self.item_features)
            assert self.number_of_items == len(self.item_features)

    def has_interactions(self) -> bool:
        return self.interactions is not None

    def has_user_features(self) -> bool:
        return self.user_features is not None

    def has_item_features(self) -> bool:
        return self.item_features is not None


def get_user_features(
    data: UserItemInteractionsDataset, user_id: int
) -> Optional[torch.Tensor]:
    if data.user_features is None:
        return None
    return data.user_features.features[user_id]


def get_item_features(
    data: UserItemInteractionsDataset, item_id: int
) -> Optional[torch.Tensor]:
    if data.item_features is None:
        return None
    return data.item_features.features[item_id]
