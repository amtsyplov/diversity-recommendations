from typing import Optional

import torch

from divrec.models.base_models import RankingModel


class RandomModel(RankingModel):
    def __init__(self, no_users: int, no_items: int):
        torch.nn.Module.__init__(self)
        self.no_users = no_users
        self.no_items = no_items

    def forward(
        self,
        user_id: torch.LongTensor,
        item_id: torch.LongTensor,
        user_features: Optional[torch.Tensor],
        item_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return torch.randn(user_id.size())
