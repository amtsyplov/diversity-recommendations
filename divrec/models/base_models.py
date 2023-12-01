from typing import Optional

import torch


class RankingModel(torch.nn.Module):
    def forward(
            self,
            user_id: torch.LongTensor,
            item_id: torch.LongTensor,
            user_features: Optional[torch.Tensor],
            item_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required \"forward\" function")
