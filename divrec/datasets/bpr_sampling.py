import random
from abc import ABC
from typing import Iterator

import torch
from torch.utils.data import IterableDataset


class BPRSampling(IterableDataset, ABC):
    def __init__(
            self,
            user_item_interactions: torch.LongTensor,
            user_item_interactions_frozen: torch.LongTensor,
            max_sampled: int,
    ):
        """

        :param user_item_interactions: torch.LongTensor with 3 columns: [user_id, item_id, interaction_score],
        usable for model fitting.

        :param user_item_interactions_frozen: torch.LongTensor with 3 columns: [user_id, item_id, interaction_score],
        usable to drop some positive examples while validation/test phase

        :param max_sampled: int, the number of triples (user_id, positive_item_id, negative_item_id) for any user.
        """
        self.frozen_interactions = user_item_interactions_frozen
        self.interactions = user_item_interactions
        self.max_sampled = max_sampled

        self.users = frozenset(torch.unique(user_item_interactions[:, 0]).tolist())
        self.items = frozenset(torch.unique(user_item_interactions[:, 1]).tolist())

    def __iter__(self) -> Iterator[int, int, int]:
        for user in self.users:
            frozen = frozenset(self.frozen_interactions[self.frozen_interactions[:, 0] == user, 1])
            positives = frozenset(self.interactions[self.interactions[:, 0] == user, 1])
            negatives = self.items - positives - frozen

            positives = random.choices(list(positives), k=self.max_sampled)
            negatives = random.choices(list(negatives), k=self.max_sampled)

            for positive, negative in zip(positives, negatives):
                yield user, positive, negative
