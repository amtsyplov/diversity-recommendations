import random
from typing import Iterator, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader, Sampler

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


PairWiseListRow = Tuple[
    int,  # user_id
    torch.LongTensor,  # positive items id
    torch.LongTensor,  # negative items id
    Optional[torch.Tensor],  # user features
    Optional[torch.Tensor],  # positive items features
    Optional[torch.Tensor],  # negative items features
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


class PairWiseListDataset(Dataset):
    """
    Gives the user_id and two list: positives and negatives
    for user. And it's features.
    """

    def __init__(
        self,
        data: UserItemInteractionsDataset,
        frozen: Optional[UserItemInteractionsDataset] = None,
        max_sampled: int = 100,
    ):
        self.data = data
        self.frozen = frozen
        self.max_sampled = max_sampled
        self.items = frozenset(range(self.data.number_of_items))

    def __len__(self):
        return self.data.number_of_users

    def __getitem__(self, user_id: int) -> PairWiseListRow:
        positives = frozenset(
            self.data.interactions[self.data.interactions[:, 0] == user_id, 1].tolist()
        )
        negatives = self.items - positives
        if self.frozen is not None:
            frozen = frozenset(
                self.frozen.interactions[
                    self.frozen.interactions[:, 0] == user_id, 1
                ].tolist()
            )
            negatives -= frozen

        positives = torch.LongTensor(list(positives))
        negatives = torch.LongTensor(list(negatives))
        return (
            user_id,
            positives,
            negatives,
            get_user_features(self.data, user_id),
            get_item_features(self.data, positives),
            get_item_features(self.data, negatives),
        )

    def loader(
        self,
        batch_size: int = 1,
        drop_last: bool = False,
        data_source=None,
        **loader_params
    ) -> DataLoader:
        assert self.max_sampled > 0 or self.frozen is None
        sampler = SameInteractionsCountSampler(
            self.data, batch_size, drop_last, data_source=data_source
        )
        return DataLoader(self, sampler=sampler, **loader_params)


class SameInteractionsCountSampler(Sampler[List[int]]):
    """
    Gives an Iterator[List[int]] via list of users
    with the same number of interactions in each list.
    """

    def __init__(
        self,
        dataset: UserItemInteractionsDataset,
        batch_size: int,
        drop_last: bool,
        data_source=None,
    ):
        Sampler.__init__(self, data_source=data_source)
        assert batch_size > 0
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.lengths = torch.LongTensor(
            [
                torch.sum(dataset.interactions[:, 0] == user).item()
                for user in range(dataset.number_of_users)
            ]
        )
        self.indexes = torch.arange(dataset.number_of_users)

    def __iter__(self):
        for length, count in torch.unique(self.lengths, return_counts=True):
            indexes = iter(self.indexes[self.lengths == length])
            batch_count = count // self.batch_size
            last_batch_size = count % self.batch_size
            for _ in range(batch_count):
                batch = [next(indexes) for _ in range(self.batch_size)]
                yield batch
            if not self.drop_last and last_batch_size > 0:
                batch = [next(indexes) for _ in range(last_batch_size)]
                yield batch


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
            positives = self.data.interactions[
                self.data.interactions[:, 0] == user_id, 1
            ]

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
