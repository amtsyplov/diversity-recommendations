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
        dataset: UserItemInteractionsDataset,
        batch_size: int = 1,
        drop_last: bool = False,
    ):
        Sampler.__init__(self)
        assert batch_size > 0
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.indexes, self.lengths = torch.unique(
            dataset.interactions[:, 0], return_counts=True
        )

    def __iter__(self):
        for length, count in zip(*torch.unique(self.lengths, return_counts=True)):
            indexes = iter(self.indexes[self.lengths == length])
            batch_count = count // self.batch_size
            last_batch_size = count % self.batch_size
            for _ in range(batch_count):
                batch = [next(indexes) for _ in range(self.batch_size)]
                yield batch
            if not self.drop_last and last_batch_size > 0:
                batch = [next(indexes) for _ in range(last_batch_size)]
                yield batch
