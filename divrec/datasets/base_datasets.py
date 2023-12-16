import random
from typing import Iterable, Iterator, List, Optional, Tuple, FrozenSet

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader

from divrec.datasets.storages import UserItemInteractionsDataset


PointWiseRow = Tuple[
    int,  # user_id
    int,  # item_id
    torch.Tensor,  # user features
    torch.Tensor,  # item features
    float,  # interactions score
]


PairWiseRow = Tuple[
    int,  # user_id
    int,  # positive item_id
    int,  # negative item_id
    torch.Tensor,  # user features
    torch.Tensor,  # positive item features
    torch.Tensor,  # negative item features
]


PairWiseListRow = Tuple[
    int,  # user_id
    torch.LongTensor,  # positive items id
    torch.LongTensor,  # negative items id
    torch.Tensor,  # user features
    torch.Tensor,  # positive items features
    torch.Tensor,  # negative items features
]


RankingRow = Tuple[
    torch.LongTensor,  # repeated user_id
    torch.LongTensor,  # positive item_id
    torch.LongTensor,  # negative item_id
    torch.Tensor,  # repeated user features
    torch.Tensor,  # negative item features
]


def sample(
    pos: Iterable[int], neg: Iterable[int], max_sampled: int = -1
) -> Tuple[List[int], List[int]]:
    positives = list(pos)
    negatives = list(neg)
    if max_sampled > 0:
        positives = random.choices(positives, k=max_sampled)
        negatives = random.choices(negatives, k=max_sampled)
    return positives, negatives


def explode(
    pos: Iterable[int], neg: Iterable[int], max_sampled: int = -1
) -> Iterator[Tuple[int, int]]:
    positives = list(pos)
    negatives = list(neg)
    if max_sampled > 0:
        positives = random.choices(positives, k=max_sampled)
        negatives = random.choices(negatives, k=max_sampled)
        for p, n in zip(positives, negatives):  # exactly max_sampled pairs for user
            yield p, n
    else:  # cartesian product of positives and negatives
        for p in positives:
            for n in negatives:
                yield p, n


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
        """

        :param data: user-item interactions for negative sampling
        :param frozen: interactions which are neither positive not negative
        :param max_sampled: if > 0 then number of random (positive, negative)
         pairs for each user per epoch; else all possible (positive, negative)
         pairs
        :param exclude_positives: if True (default) positives and negatives
        intersection will be empty; else positives may be subset of negatives
        for each user
        """
        self.data = data
        self.frozen = frozen
        self.max_sampled = max_sampled
        self.exclude_positives = exclude_positives
        self.items = frozenset(range(self.data.number_of_items))

    def get_frozen(self, user_id: int) -> FrozenSet[int]:
        if self.frozen is None:
            return frozenset()
        return frozenset(self.frozen.get_user_interactions(user_id).tolist())

    def get_positives(self, user_id: int) -> FrozenSet[int]:
        return frozenset(self.data.get_user_interactions(user_id).tolist())

    def get_negatives(
        self, user_id: int, pos: Optional[FrozenSet[int]] = None
    ) -> FrozenSet[int]:
        """
        If exclude_positives is True, then excludes positive items from negatives.
        Usable for validation mode when positives are needed as a separated list
        and as a part of the list of items to score. Train mode by default.
        """
        frozen = self.get_frozen(user_id)
        if self.exclude_positives:
            positives = pos if pos is not None else self.get_positives(user_id)
            return self.items - positives - frozen
        return self.items - frozen


class PointWiseDataset(Dataset):
    def __init__(self, data: UserItemInteractionsDataset):
        self.data = data

    def __len__(self) -> int:
        return self.data.number_of_interactions

    def __getitem__(self, item: int) -> PointWiseRow:
        user_id, item_id = self.data.interactions[item]
        user_features = self.data.get_user_features(user_id)
        item_features = self.data.get_item_features(item_id)
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

        pos_users, pos_counts = torch.unique(
            self.data.interactions[:, 0], return_counts=True
        )
        frozen_users, frozen_counts = torch.unique(
            self.frozen.interactions[:, 0], return_counts=True
        )

        counts = torch.zeros(self.data.number_of_users)
        counts[pos_users] += pos_counts
        counts[frozen_users] -= frozen_counts

        return torch.sum(
            counts[pos_users] * (self.data.number_of_items - counts[pos_users])
        ).item()

    def __getitem__(self, item):
        raise NotImplementedError("__getitem__ is not available for IterableDataset")

    def __iter__(self) -> Iterator[PairWiseRow]:
        for user_id in range(self.data.number_of_users):
            user_features = self.data.get_user_features(user_id)
            positives = self.get_positives(user_id)
            negatives = self.get_negatives(user_id, pos=positives)
            for pos, neg in explode(positives, negatives, max_sampled=self.max_sampled):
                pos_features = self.data.get_item_features(pos)
                neg_features = self.data.get_item_features(neg)
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
        positives, negatives = sample(
            positives, negatives, max_sampled=self.max_sampled
        )
        return (
            user_id,
            torch.LongTensor(positives),
            torch.LongTensor(negatives),
            self.data.get_user_features(user_id),
            self.data.get_item_features(positives),
            self.data.get_item_features(negatives),
        )


class RankingDataset(IterableDataset, PairWiseLists):
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
        PairWiseLists.__init__(
            self,
            data,
            frozen=frozen,
            max_sampled=-1,
            exclude_positives=False,
        )

    def __getitem__(self, user_id: int) -> RankingRow:
        positives = torch.LongTensor(list(self.get_positives(user_id)))

        negatives = torch.LongTensor(
            list(self.get_negatives(user_id))
        )  # all items except frozen
        neg_features = self.data.get_item_features(negatives)

        repeated_user_id = torch.LongTensor([user_id for _ in negatives])
        repeated_user_features = self.data.get_user_features(repeated_user_id)

        yield (
            repeated_user_id,
            positives,
            negatives,
            repeated_user_features,
            neg_features,
        )

    def __len__(self):
        return self.data.number_of_users

    def __iter__(self) -> Iterator[RankingRow]:
        for user_id in range(self.data.number_of_users):
            yield self[user_id]
