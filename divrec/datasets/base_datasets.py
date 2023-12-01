import random
from typing import Iterator, Optional, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset

from divrec.datasets.storages import UserItemInteractionsDataset, get_item_features, get_user_features


PointWiseRow = Tuple[
    int,  # user_id
    int,  # item_id
    Optional[torch.Tensor],  # user features
    Optional[torch.Tensor],  # item features
    float  # interactions score
]


PairWiseRow = Tuple[
    int,  # user_id
    int,  # positive item_id
    int,  # negative item_id
    Optional[torch.Tensor],  # user features
    Optional[torch.Tensor],  # positive item features
    Optional[torch.Tensor],  # negative item features
]


class PointWiseDataset(Dataset):
    def __init__(self, data: UserItemInteractionsDataset):
        self.data = data

    def __len__(self) -> int:
        return self.data.number_of_interactions

    def __getitem__(self, item: int) -> PointWiseRow:
        user_id, item_id = self.data.interactions[item]
        user_features = get_user_features(self.data, user_id)
        item_features = get_item_features(self.data, item_id)
        interaction_score = self.data.interaction_scores[item_id]
        return user_id, item_id, user_features, item_features, interaction_score


class PairWiseDataset(IterableDataset):
    def __init__(
            self,
            data: UserItemInteractionsDataset,
            frozen: UserItemInteractionsDataset,
            max_sampled: int
    ):
        self.data = data
        self.frozen = frozen
        self.max_sampled = max_sampled

    def __iter__(self) -> Iterator[PairWiseRow]:
        items = frozenset(range(self.data.number_of_items))
        for user_id in range(self.data.number_of_users):
            user_features = get_user_features(self.data, user_id)
            positives = frozenset(self.data.interactions[self.data.interactions[:, 0] == user_id, 1].tolist())
            frozen = frozenset(self.frozen.interactions[self.frozen.interactions[:, 0] == user_id, 1].tolist())
            negatives = items - positives - frozen

            if self.max_sampled > 0:
                positives = random.choices(list(positives), k=self.max_sampled)
                negatives = random.choices(list(negatives), k=self.max_sampled)

            for positive_item_id in positives:
                positive_item_features = get_item_features(self.data, positive_item_id)
                for negative_item_id in negatives:
                    negative_item_features = get_item_features(self.data, negative_item_id)
                    yield (
                        user_id,
                        positive_item_id,
                        negative_item_id,
                        user_features,
                        positive_item_features,
                        negative_item_features,
                    )
