import os
from typing import Dict, Any

import pandas as pd

import torch

from divrec.datasets import UserItemInteractionsDataset
from divrec_pipelines.pipeline import Container, Stage


class LoadMovieLens100K(Stage):
    """Load train test split for MovieLens100K"""

    @property
    def name(self) -> str:
        return "load_movie_lens_100_k"

    def call(self, config: Dict[str, Any], arg: Container) -> Container:
        info = self.load_info("u.info")
        data = self.load_data("u.data")
        base = self.load_data("ua.base")
        test = self.load_data("ua.test")

        base.number_of_users = info["users"]
        base.number_of_items = info["items"]

        test.number_of_users = info["users"]
        test.number_of_items = info["items"]

        return Container({"info": info, "data": data, "base": base, "test": test,})

    def load_data(self, filename: str) -> UserItemInteractionsDataset:
        data = pd.read_csv(
            os.path.join(self.config["data_dir"], filename),
            sep="\t",
            names=["user_id", "item_id", "score"],
            usecols=[0, 1, 2],
        )

        # interactions = interactions - 1 because of starting from 1 in actual data
        interactions = torch.LongTensor(data[["user_id", "item_id"]].values) - 1
        interaction_scores = torch.Tensor(data["score"].values)
        return UserItemInteractionsDataset(
            interactions=interactions, interaction_scores=interaction_scores
        )

    def load_info(self, filename: str) -> Dict[str, int]:
        info = {}
        with open(os.path.join(self.config["data_dir"], filename), mode="r") as file:
            for line in file.readlines():
                v, k = line.split()
                info[k] = int(v)
        return info


configuration = {"data_dir": "data/"}

load_movie_lens_100_k = LoadMovieLens100K(configuration)
