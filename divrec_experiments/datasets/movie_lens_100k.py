import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass
class MovieLens100K:
    no_users: int
    no_items: int
    no_ratings: int
    train: torch.Tensor
    test: torch.Tensor
    validation: torch.Tensor


def movie_lens_load(path: str, train_size: float, test_size: float) -> MovieLens100K:
    """
    Loads MovieLens100K dataset, split it to train, validation and test
    in time order and return tuple of 3 tensors in form

    [user_id, item_id, score]

    Stratified by user_id.
    """
    info = {}
    with open(os.path.join(path, "u.info"), mode="r") as file:
        for line in file.readlines():
            v, k = line.split()
            info["no_" + k] = int(v)

    rating = pd.read_csv("ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])\
        .sort_values("timestamp", ignore_index=True)

    # in initial data numerations starts from 1
    rating["item_id"] = rating["item_id"] - 1
    rating["user_id"] = rating["user_id"] - 1

    users, indexes, counts = np.unique(rating["user_id"], return_inverse=True, return_counts=True)

    rating["counts"] = counts[indexes]
    rating["row_number"] = rating.groupby("user_id").cumcount()
    rating["ratio"] = rating["row_number"] / rating["counts"]

    rating["sample"] = np.where(
        rating["ratio"] < train_size, "train",
        np.where(rating["ratio"] < (1 - test_size), "validation", "test")
    )

    train = rating.loc[rating["sample"] == "train", ["user_id", "item_id", "rating"]].values
    test = rating.loc[rating["sample"] == "test", ["user_id", "item_id", "rating"]].values
    validation = rating.loc[rating["sample"] == "validation", ["user_id", "item_id", "rating"]].values

    return MovieLens100K(
        info["no_users"],
        info["no_items"],
        info["no_ratings"],
        torch.LongTensor(train),
        torch.LongTensor(test),
        torch.LongTensor(validation),
    )
