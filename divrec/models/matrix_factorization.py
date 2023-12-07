from typing import Optional

import torch

from divrec.models.base_models import RankingModel


class MatrixFactorization(RankingModel):
    def __init__(self, no_users: int, no_items: int, embedding_dim: int):
        torch.nn.Module.__init__(self)

        self.no_users = no_users
        self.no_items = no_items
        self.embedding_dim = embedding_dim

        self.user_embeddings = torch.nn.Embedding(no_users, embedding_dim)
        self.item_embeddings = torch.nn.Embedding(no_items, embedding_dim)

    def forward(
        self,
        user_id: torch.LongTensor,
        item_id: torch.LongTensor,
        user_features: Optional[torch.Tensor] = None,
        item_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        u = self.user_embeddings(user_id)
        i = self.item_embeddings(item_id)
        return torch.sum(u * i, dim=1)
