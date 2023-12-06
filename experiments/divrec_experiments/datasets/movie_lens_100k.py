import os
from dataclasses import dataclass

import pandas as pd
import torch

from divrec.datasets import UserItemInteractionsDataset, Features


@dataclass
class MovieLens100K:
    train: UserItemInteractionsDataset
    test: UserItemInteractionsDataset


def load_data(path: str, filename: str) -> UserItemInteractionsDataset:
    info = {}
    with open(os.path.join(path, "u.info"), mode="r") as file:
        for line in file.readlines():
            v, k = line.split()
            info["no_" + k] = int(v)

    rating = pd.read_csv(
        os.path.join(path, filename),
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    ).sort_values("timestamp", ignore_index=True)
    rating["user_id"] = rating["user_id"] - 1
    rating["item_id"] = rating["item_id"] - 1

    user_features = pd.read_csv(
        os.path.join(path, "u.user"),
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
    )
    user_features.drop(
        columns=["user_id", "gender", "occupation", "zip_code"],
        inplace=True,
    )

    item_features = pd.read_csv(
        os.path.join(path, "u.item"),
        encoding="latin-1",
        sep="|",
        names=[
            "movie_id",
            "movie_title",
            "release_date",
            "video_release_date",
            "IMDb_URL",
            "unknown",
            "Action",
            "Adventure",
            "Animation",
            "Children's",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ],
    )
    item_features.drop(
        columns=[
            "movie_id",
            "movie_title",
            "release_date",
            "video_release_date",
            "IMDb_URL",
        ],
        inplace=True,
    )

    return UserItemInteractionsDataset(
        interactions=torch.LongTensor(rating[["user_id", "item_id"]].values),
        interaction_scores=torch.Tensor(rating["rating"].values),
        item_features=Features(
            torch.Tensor(item_features.values.tolist()), item_features.columns
        ),
        user_features=Features(
            torch.Tensor(user_features.values.tolist()), user_features.columns
        ),
        number_of_items=info["no_items"],
        number_of_users=info["no_users"],
    )


def movie_lens_load(path: str) -> MovieLens100K:
    return MovieLens100K(
        load_data(path, "ua.base"),
        load_data(path, "ua.test"),
    )
