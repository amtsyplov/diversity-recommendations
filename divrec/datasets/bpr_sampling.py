import random
from abc import ABC

import torch
from torch.utils.data import IterableDataset


class BPRSampling(IterableDataset, ABC):
    def __init__(
            self,
            user_item_interactions: torch.LongTensor,
            max_sampled: int,
    ):
        self.interactions = user_item_interactions
        self.max_sampled = max_sampled

        self.users = frozenset(torch.unique(user_item_interactions[:, 0]).tolist())
        self.items = frozenset(torch.unique(user_item_interactions[:, 1]).tolist())

    def __iter__(self):
        for user in self.users:
            positives = frozenset(self.interactions[self.interactions[:, 0] == user, 1])
            negatives = self.items - positives

            positives = random.choices(list(positives), k=self.max_sampled)
            negatives = random.choices(list(negatives), k=self.max_sampled)

            for positive, negative in zip(positives, negatives):
                yield user, positive, negative
