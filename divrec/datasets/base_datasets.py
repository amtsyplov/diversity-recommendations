import random
from typing import Iterator, List, Optional, Tuple, FrozenSet

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader

from divrec.datasets.storages import (
    UserItemInteractionsDataset,
    get_item_features,
    get_user_features,
    get_user_interactions,
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


class PairWiseLists:
    """Special class to work with pairwise
    datasets in different forms"""
    def __init__(
        self,
        data: UserItemInteractionsDataset,
        frozen: Optional[UserItemInteractionsDataset] = None,
        max_sampled: int = 100,
        exclude_positives: bool = True,
    ):
        self.data = data
        self.frozen = frozen
        self.max_sampled = max_sampled
        self.exclude_positives = exclude_positives
        self.items = frozenset(range(self.data.number_of_items))

    def get_frozen(self, user_id: int) -> FrozenSet[int]:
        if self.frozen is None:
            return frozenset()
        return frozenset(get_user_interactions(self.frozen, user_id).tolist())

    def get_positives(self, user_id: int) -> FrozenSet[int]:
        return frozenset(get_user_interactions(self.data, user_id).tolist())

    def get_negatives(self, user_id: int, pos: Optional[FrozenSet[int]] = None) -> FrozenSet[int]:
        positives = frozenset() if pos is None or self.exclude_positives else pos
        frozen = self.get_frozen(user_id)
        return self.items - positives - frozen

    def sample(self, pos: FrozenSet[int], neg: FrozenSet[int]) -> Tuple[List[int], List[int]]:
        positives = list(pos)
        negatives = list(neg)
        if self.max_sampled > 0:
            positives = random.choices(positives, k=self.max_sampled)
            negatives = random.choices(negatives, k=self.max_sampled)
        return positives, negatives

    def explode(self, pos: FrozenSet[int], neg: FrozenSet[int]) -> Iterator[Tuple[int, int]]:
        positives, negatives = self.sample(pos, neg)
        if self.max_sampled > 0:
            for p, n in zip(positives, negatives):
                yield p, n
        else:
            for p in positives:
                for n in negatives:
                    yield p, n


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


class PairWiseDataset(IterableDataset, PairWiseLists):
    def __init__(self, *args, **kwargs):
        PairWiseLists.__init__(self, *args, **kwargs)

    def __len__(self) -> int:
        if self.max_sampled > 0:
            return self.data.number_of_users * self.max_sampled
        _, counts = torch.unique(self.data.interactions[:, 0], return_counts=True)
        return torch.sum(counts * (self.data.number_of_items - counts)).item()

    def __getitem__(self, item):
        raise NotImplementedError("__getitem__ is not available for IterableDataset")

    def __iter__(self) -> Iterator[PairWiseRow]:
        for user_id in range(self.data.number_of_users):
            user_features = get_user_features(self.data, user_id)
            positives = self.get_positives(user_id)
            negatives = self.get_negatives(user_id, pos=positives)
            for pos, neg in self.explode(positives, negatives):
                pos_features = get_item_features(self.data, pos)
                neg_features = get_item_features(self.data, neg)
                yield (
                    user_id,
                    pos,
                    neg,
                    user_features,
                    pos_features,
                    neg_features,
                )

    def loader(self, **loader_params) -> DataLoader:
        return DataLoader(self, **loader_params)


class PairWiseListDataset(Dataset, PairWiseLists):
    """
    Gives the user_id and two list: positives and negatives
    for user. And it's features.
    """
    def __init__(self, *args, **kwargs):
        PairWiseLists.__init__(self, *args, **kwargs)

    def __len__(self):
        return self.data.number_of_users

    def __getitem__(self, user_id: int) -> PairWiseListRow:
        positives = self.get_positives(user_id)
        negatives = self.get_negatives(user_id, pos=positives)
        positives, negatives = self.sample(positives, negatives)
        positives, negatives = torch.LongTensor(positives), torch.LongTensor(negatives)
        return (
            user_id,
            positives,
            negatives,
            get_user_features(self.data, user_id),
            get_item_features(self.data, positives),
            get_item_features(self.data, negatives),
        )


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
