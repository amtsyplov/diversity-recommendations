from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from divrec.datasets import UserItemInteractionsDataset
from divrec.loaders import BPRSampling
from divrec.losses import LogSigmoidDifferenceLoss, IntraListBinaryUnfairnessScore
from divrec.models import MatrixFactorization

from divrec_pipelines.pipeline import Stage, Container


class TrainModel(Stage):
    @property
    def name(self) -> str:
        return "train_model"

    def call(self, config: Dict[str, Any], arg: Container) -> Container:
        dataset: UserItemInteractionsDataset = arg.elements["base"]
        bpr_dataset = BPRSampling(
            dataset.interactions, max_sampled=config["max_sampled"]
        )
        data_loader = DataLoader(bpr_dataset, batch_size=config["batch_size"])

        model = MatrixFactorization(
            no_users=arg.elements["info"]["users"],
            no_items=arg.elements["info"]["items"],
            embedding_dim=config["embedding_dim"],
        )
        model.to("cpu")

        bpr_loss = LogSigmoidDifferenceLoss()
        ilbu_regularization = IntraListBinaryUnfairnessScore(dataset=dataset)

        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

        return Container()
