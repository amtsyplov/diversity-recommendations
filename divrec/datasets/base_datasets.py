import random
from typing import Iterator, Optional, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader

from divrec.datasets.storages import (
    UserItemInteractionsDataset,
    get_item_features,
    get_user_features,
)


PointWiseRow = Tuple[
    int,  # user_id
    int,  # item_id
    Optional[torch.Tensor],  # user features
    Optional[torch.Tensor],  # item features
    float,  # interactions score
]


PairWiseRow = Tuple[
    int,  # user_id
    int,  # positive item_id
    int,  # negative item_id
    Optional[torch.Tensor],  # user features
    Optional[torch.Tensor],  # positive item features
    Optional[torch.Tensor],  # negative item features
]


RankingRow = Tuple[
    torch.LongTensor,  # repeated user_id
    torch.LongTensor,  # positive item_id
    torch.LongTensor,  # negative item_id
    Optional[torch.Tensor],  # repeated user features
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

    def loader(self, **loader_params) -> DataLoader:
        return DataLoader(self, **loader_params)


class PairWiseDataset(IterableDataset):
    def __init__(
        self,
        data: UserItemInteractionsDataset,
        frozen: Optional[UserItemInteractionsDataset] = None,
        max_sampled: int = 100,
    ):
        self.data = data
        self.frozen = frozen
        self.max_sampled = max_sampled

    def __iter__(self) -> Iterator[PairWiseRow]:
        items = frozenset(range(self.data.number_of_items))
        for user_id in range(self.data.number_of_users):
            user_features = get_user_features(self.data, user_id)
            positives = frozenset(
                self.data.interactions[
                    self.data.interactions[:, 0] == user_id, 1
                ].tolist()
            )

            if self.frozen is not None:
                frozen = frozenset(
                    self.frozen.interactions[
                        self.frozen.interactions[:, 0] == user_id, 1
                    ].tolist()
                )
                negatives = items - positives - frozen
            else:
                negatives = items - positives

            if self.max_sampled > 0:
                positives = random.choices(list(positives), k=self.max_sampled)
                negatives = random.choices(list(negatives), k=self.max_sampled)

            for positive_item_id in positives:
                positive_item_features = get_item_features(self.data, positive_item_id)
                for negative_item_id in negatives:
                    negative_item_features = get_item_features(
                        self.data, negative_item_id
                    )
                    yield (
                        user_id,
                        positive_item_id,
                        negative_item_id,
                        user_features,
                        positive_item_features,
                        negative_item_features,
                    )

    def loader(self, **loader_params) -> DataLoader:
        return DataLoader(self, **loader_params)


class RankingDataset(IterableDataset):
    """
    Maybe the most difficult to understand dataset.
    Has no loader method. In one iteration gives
    negative items for one user.

    Gives user_id repeated as a torch.LongTensor and
    len(user_id) == len(negatives)

    Gives user_features repeated as a torch.Tensor and
    len(user_features) == len(negatives)

    While building negatives remove only frozen interactions.
    """

    def __init__(
        self,
        data: UserItemInteractionsDataset,
        frozen: Optional[UserItemInteractionsDataset] = None,
    ):
        self.data = data
        self.frozen = frozen

    def __iter__(self) -> Iterator[RankingRow]:
        items = frozenset(range(self.data.number_of_items))
        for user_id in range(self.data.number_of_users):
            positives = self.data.interactions[self.data.interactions[:, 0] == user_id, 1]

            if self.frozen is not None:
                frozen = frozenset(
                    self.frozen.interactions[
                        self.frozen.interactions[:, 0] == user_id, 1
                    ].tolist()
                )
                negatives = list(items - frozen)
            else:
                negatives = items

            user_features = None
            if self.data.has_user_features():
                user_features = torch.concatenate(
                    [self.data.user_features.features for _ in negatives]
                )

            item_features = None
            if self.data.has_user_features():
                item_features = torch.concatenate(
                    [get_item_features(self.data, item_id) for item_id in negatives]
                )

            yield (
                torch.full((len(negatives),), user_id),
                positives,
                torch.LongTensor(negatives),
                user_features,
                item_features,
            )
