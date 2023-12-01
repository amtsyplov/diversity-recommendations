from copy import deepcopy
from typing import Dict, Any

import torch

from divrec.datasets import Features
from divrec_pipelines.pipeline import Container, stage


IS_POPULAR_FEATURE_NAME = "is_popular"


@stage(
    configuration={"quantile": 0.8,}
)
def add_popularity_feature(config: Dict[str, Any], arg: Container) -> Container:
    outputs = deepcopy(arg.elements)
    interactions = outputs.pop("data").interactions

    _, counts = torch.unique(interactions[:, 1], return_counts=True)
    q = torch.quantile(counts.float(), config["quantile"])
    popularity_labels = (counts > q).int().unsqueeze(1)
    feature = Features(
        features=popularity_labels, feature_names=[IS_POPULAR_FEATURE_NAME]
    )

    outputs["base"].item_features = feature
    outputs["test"].item_features = feature
    return Container(elements=outputs)
