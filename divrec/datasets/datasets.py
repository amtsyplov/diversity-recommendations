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
        self.number_of_features = self.features.size(0)
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

            users = torch.unique(self.interactions[:, 0])
            items = torch.unique(self.interactions[:, 1])

            self.number_of_users = max(self.number_of_users, users.size(0))
            self.number_of_items = max(self.number_of_items, items.size(0))

            assert torch.all(torch.arange(self.number_of_users) == users).item()
            assert torch.all(torch.arange(self.number_of_items) == items).item()

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
