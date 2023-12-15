from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch


@dataclass
class Features:
    """
    Abstraction for table with named columns
    """

    features: Optional[torch.Tensor] = torch.empty((0, 0), dtype=torch.double)
    feature_names: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.features.size(1) == 0:
            self.features = torch.empty(
                (0, len(self.feature_names)), dtype=torch.double
            )
        assert self.features.size(1) == len(self.feature_names)

    def __len__(self) -> int:
        """The number of features"""
        return len(self.feature_names)

    def __getitem__(
        self, item: Union[str, List[str], int, torch.LongTensor]
    ) -> torch.Tensor:
        if isinstance(item, str):  # feature name
            return self.features[:, self.feature_names.index(item)]
        elif isinstance(item, list) and all(
            isinstance(name, str) for name in item
        ):  # features list
            columns = torch.LongTensor(
                [self.feature_names.index(name) for name in item]
            )
            return self.features[:, columns]
        return self.features[item]  # user or item id or LongTensor of user or item ids

    def __contains__(self, item: Union[int, str]) -> bool:
        if isinstance(item, str):
            return item in self.feature_names
        return self.features.size(0) > item


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
    user_features: Features = field(default_factory=Features)
    item_features: Features = field(default_factory=Features)

    number_of_interactions: int = 0
    number_of_users: int = 0
    number_of_items: int = 0

    def __post_init__(self) -> None:
        if self.interactions is not None:
            self.number_of_interactions, no_columns = self.interactions.size()
            assert no_columns == 2

            if self.interaction_scores is None:
                self.interaction_scores = torch.ones(self.number_of_interactions)
            assert self.interaction_scores.size(0) == self.number_of_interactions

            users = torch.unique(self.interactions[:, 0])
            items = torch.unique(self.interactions[:, 1])

            self.number_of_users = max(self.number_of_users, users.max().item() + 1)
            self.number_of_items = max(self.number_of_items, items.max().item() + 1)

            assert torch.all(
                torch.isin(users, torch.arange(self.number_of_users))
            ).item()
            assert torch.all(
                torch.isin(items, torch.arange(self.number_of_items))
            ).item()

        if self.interaction_scores is not None:
            assert self.interactions is not None
            assert self.interaction_scores.size(0) == self.number_of_interactions

        if self.number_of_users == 0:
            self.number_of_users = len(self.user_features)
        assert self.number_of_users == len(self.user_features)

        if self.number_of_items == 0:
            self.number_of_items = len(self.item_features)
        assert self.number_of_items == len(self.item_features)

    def has_interactions(self) -> bool:
        return self.interactions is not None

    def get_user_features(self, user_id: Union[int, torch.LongTensor]) -> torch.Tensor:
        return self.user_features[user_id]

    def get_item_features(self, item_id: Union[int, torch.LongTensor]) -> torch.Tensor:
        return self.item_features[item_id]

    def get_user_interactions(
        self, user_id: Union[int, torch.LongTensor]
    ) -> Optional[torch.LongTensor]:
        if self.interactions is None:
            return None
        elif isinstance(user_id, int):
            return self.interactions[self.interactions[:, 0] == user_id, 1]
        return self.interactions[torch.isin(self.interactions[:, 0], user_id), 1]

    def get_item_interactions(
        self, item_id: Union[int, torch.LongTensor]
    ) -> Optional[torch.LongTensor]:
        if self.interactions is None:
            return None
        elif isinstance(item_id, int):
            return self.interactions[self.interactions[:, 1] == item_id, 0]
        return self.interactions[torch.isin(self.interactions[:, 1], item_id), 0]
