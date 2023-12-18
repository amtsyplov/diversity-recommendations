from typing import Tuple

import torch


PointWiseRow = Tuple[
    int,  # user_id
    int,  # item_id
    torch.Tensor,  # user features, size = (no_user_features,)
    torch.Tensor,  # item features, size = (no_item_features,)
    float,  # interactions score
]


PointWiseBatch = Tuple[
    torch.LongTensor,  # user_id, size = (batch_size,)
    torch.LongTensor,  # item_id, size = (batch_size,)
    torch.Tensor,  # user features, size = (batch_size, no_user_features)
    torch.Tensor,  # item features, size = (batch_size, no_item_features)
    torch.Tensor,  # interactions score, size = (batch_size,)
]


PairWiseRow = Tuple[
    int,  # user_id
    int,  # positive item_id
    int,  # negative item_id
    torch.Tensor,  # user features, size = (no_user_features,)
    torch.Tensor,  # positive item features, size = (no_item_features,)
    torch.Tensor,  # negative item features, size = (no_item_features,)
]


PairWiseBatch = Tuple[
    torch.LongTensor,  # user_id, size = (batch_size,)
    torch.LongTensor,  # positive item_id, size = (batch_size,)
    torch.LongTensor,  # negative item_id, size = (batch_size,)
    torch.Tensor,  # user features, size = (batch_size, no_user_features)
    torch.Tensor,  # positive item features, size = (batch_size, no_item_features)
    torch.Tensor,  # negative item features, size = (batch_size, no_item_features)
]


PairWiseListRow = Tuple[
    int,  # user_id
    torch.LongTensor,  # positive items id
    torch.LongTensor,  # negative items id
    torch.Tensor,  # user features
    torch.Tensor,  # positive items features
    torch.Tensor,  # negative items features
]


PairWiseListBatch = Tuple[
    torch.LongTensor,  # user_id, size = (batch_size,)
    torch.LongTensor,  # positive items id, size = (batch_size, no_positive_items)
    torch.LongTensor,  # negative items id, size = (batch_size, no_negative_items)
    torch.Tensor,  # user features, size = (batch_size, no_user_features)
    torch.Tensor,  # positive items features, size = (batch_size, no_positive_items, no_item_features)
    torch.Tensor,  # negative items features, size = (batch_size, no_negative_items, no_item_features)
]


RankingRow = Tuple[
    torch.LongTensor,  # repeated user_id
    torch.LongTensor,  # positive item_id
    torch.LongTensor,  # negative item_id
    torch.Tensor,  # repeated user features
    torch.Tensor,  # negative item features
]
