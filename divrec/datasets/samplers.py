from typing import Iterator, List, Optional

import torch
from torch.utils.data import Sampler

from divrec.datasets.storages import UserItemInteractionsDataset


class SameInteractionsCountSampler(Sampler):
    """
    Gives an Iterator[List[int]] via list of users
    with the same number of interactions in each list.
    """

    def __init__(
        self,
        data: UserItemInteractionsDataset,
        frozen: Optional[UserItemInteractionsDataset] = None,
        batch_size: int = 1,
        drop_last: bool = False,
    ):
        Sampler.__init__(self)
        assert batch_size > 0
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.users = torch.arange(data.number_of_users, dtype=torch.long)
        self.positive_counts = torch.zeros(data.number_of_users, dtype=torch.long)
        self.negative_counts = torch.ones(data.number_of_users, dtype=torch.long) * data.number_of_items

        data_users, data_counts = torch.unique(data.interactions[:, 0], return_counts=True)
        self.positive_counts[data_users] += data_counts
        self.negative_counts[data_users] -= data_counts

        if frozen is not None:
            frozen_users, frozen_counts = torch.unique(frozen.interactions[:, 0], return_counts=True)
            self.negative_counts[frozen_users] -= frozen_counts

    def __iter__(self) -> Iterator[List[int]]:
        unique_positive_counts = torch.unique(self.positive_counts)
        unique_negative_counts = torch.unique(self.negative_counts)

        for pos in unique_positive_counts:
            for neg in unique_negative_counts:
                mask = (self.positive_counts == pos) & (self.negative_counts == neg)
                indexes = self.users[mask].tolist()

                indexes_iterator = iter(indexes)
                batch_count = len(indexes) // self.batch_size

                for _ in range(batch_count):
                    batch = [next(indexes_iterator) for _ in range(self.batch_size)]
                    yield batch

                if not self.drop_last:
                    last_batch_size = len(indexes) % self.batch_size
                    if last_batch_size > 0:
                        last_batch = [next(indexes_iterator) for _ in range(last_batch_size)]
                        yield last_batch
