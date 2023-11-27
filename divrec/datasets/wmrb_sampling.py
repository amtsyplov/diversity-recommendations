from abc import ABC

import torch
from torch.utils.data import IterableDataset


class WMRBSampling(IterableDataset, ABC):
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

        :param max_sampled: int, the number of sampled elements for rank estimation
        """
        self.frozen_interactions = user_item_interactions_frozen
        self.interactions = user_item_interactions
        self.max_sampled = max_sampled

        self.users = torch.unique(user_item_interactions[:, 0])
        self.items = torch.unique(user_item_interactions[:, 1])

        self.no_users = self.users.size(0)
        self.no_items = self.items.size(0)

    def __iter__(self):
        for user in self.users:
            frozen = self.frozen_interactions[self.frozen_interactions[:, 0] == user, 1]
            positives = self.interactions[self.interactions[:, 0] == user, 1]

            for positive in positives:
                sampled_indices = torch.randint(0, self.no_items, (self.max_sampled,))
                sampled = self.items[sampled_indices]
                mask = torch.isin(sampled, positives) | torch.isin(sampled, frozen)
                yield user, positive, sampled, ~mask
