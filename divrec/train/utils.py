from typing import List, Optional, Tuple

import torch

from divrec.datasets import PairWiseDataset, PointWiseDataset, RankingDataset
from divrec.losses import PointWiseLoss, PairWiseLoss, RecommendationsAwareLoss
from divrec.models import RankingModel


def point_wise_score_loop(
    dataset: PointWiseDataset,
    model: RankingModel,
    losses: List[PointWiseLoss],
    **loader_params,
):
    model.eval()
    loader = dataset.loader(**loader_params)
    loss_values = {loss.__class__.__name__: list() for loss in losses}
    for user_id, item_id, user_features, item_features, true_relevance in loader:
        predicted_relevance = model(user_id, item_id, user_features, item_features)
        for loss in losses:
            loss_values[loss.__class__.__name__].append(
                loss.point_wise(true_relevance, predicted_relevance)
            )
    return [
        loss.reduce_loss_values(torch.concatenate(loss_values[loss.__class__.__name__]))
        for loss in losses
    ]


def pair_wise_score_loop(
    dataset: PairWiseDataset,
    model: RankingModel,
    losses: List[PairWiseLoss],
    **loader_params,
):
    model.eval()
    loader = dataset.loader(**loader_params)
    loss_values = {loss.__class__.__name__: list() for loss in losses}
    for user_id, pos, neg, user_features, pos_features, neg_features in loader:
        positives = model(user_id, pos, user_features, pos_features)
        negatives = model(user_id, neg, user_features, neg_features)
        for loss in losses:
            loss_values[loss.__class__.__name__].append(
                loss.pair_wise(positives, negatives)
            )
    return [
        loss.reduce_loss_values(torch.concatenate(loss_values[loss.__class__.__name__]))
        for loss in losses
    ]


def get_model_recommendations(
    dataset: RankingDataset, model: RankingModel, number_of_recommendations: int,
) -> torch.LongTensor:
    recommendations = []
    for (
        repeated_user_id,
        positive_item_id,
        negative_item_id,
        repeated_user_features,
        negative_item_features,
    ) in dataset:
        model_scores = model(
            repeated_user_id,
            negative_item_id,
            repeated_user_features,
            negative_item_features,
        )
        recommendations.append(
            negative_item_id[torch.argsort(model_scores)][
                :number_of_recommendations
            ].tolist()
        )
    return torch.LongTensor(recommendations)


def recommendations_score_loop(
    dataset: RankingDataset,
    model: RankingModel,
    losses: List[RecommendationsAwareLoss],
    number_of_recommendations: int,
):
    model.eval()
    interactions = dataset.data.interactions
    recommendations = get_model_recommendations(
        dataset, model, number_of_recommendations
    )
    return [loss(interactions, recommendations) for loss in losses]


def point_wise_train_loop(
    dataset: PointWiseDataset,
    model: RankingModel,
    loss: PointWiseLoss,
    optimizer: torch.optim.Optimizer,
    scores: Optional[List[PointWiseLoss]] = None,
    **loader_params,
) -> Tuple[float, List[float]]:
    assert all(score.reduce for score in scores)
    model.train()
    loader = dataset.loader(**loader_params)
    batch_count = 0
    mean_loss = 0.0
    mean_scores = [0.0 for _ in range(len(scores))] if scores is not None else list()
    for user_id, item_id, user_features, item_features, true_relevance in loader:
        predicted_relevance = model(user_id, item_id, user_features, item_features)

        loss_value = loss(true_relevance, predicted_relevance)
        loss_value.backward()

        optimizer.step()
        optimizer.zero_grad()

        batch_count += 1
        mean_loss += loss_value.item()

        if scores is not None:
            for i, score in enumerate(scores):
                mean_scores[i] += score(true_relevance, predicted_relevance).item()

    return (
        mean_loss / batch_count,
        [mean_score / batch_count for mean_score in mean_scores],
    )


def pair_wise_train_loop(
    dataset: PairWiseDataset,
    model: RankingModel,
    loss: PairWiseLoss,
    optimizer: torch.optim.Optimizer,
    scores: Optional[List[PairWiseLoss]] = None,
    **loader_params,
) -> Tuple[float, List[float]]:
    assert all(score.reduce for score in scores)
    model.train()
    loader = dataset.loader(**loader_params)
    batch_count = 0
    mean_loss = 0.0
    mean_scores = [0.0 for _ in range(len(scores))] if scores is not None else list()
    for user_id, pos, neg, user_features, pos_features, neg_features in loader:
        positives = model(user_id, pos, user_features, pos_features)
        negatives = model(user_id, neg, user_features, neg_features)

        loss_value = loss(positives, negatives)
        loss_value.backward()

        optimizer.step()
        optimizer.zero_grad()

        batch_count += 1
        mean_loss += loss_value.item()

        if scores is not None:
            for i, score in enumerate(scores):
                mean_scores[i] += score(positives, negatives).item()

    return (
        mean_loss / batch_count,
        [mean_score / batch_count for mean_score in mean_scores],
    )


def recommendations_train_loop(
    dataset: RankingDataset,
    model: RankingModel,
    loss: RecommendationsAwareLoss,
    number_of_recommendations: int,
    optimizer: torch.optim.Optimizer,
    scores: Optional[List[RecommendationsAwareLoss]] = None,
) -> Tuple[float, List[float]]:
    assert all(score.reduce for score in scores)
    model.train()
    batch_count = 0
    mean_loss = 0.0
    mean_scores = [0.0 for _ in range(len(scores))] if scores is not None else list()
    for (
        repeated_user_id,
        positive_item_id,
        negative_item_id,
        repeated_user_features,
        negative_item_features,
    ) in dataset:
        model_scores = model(
            repeated_user_id,
            negative_item_id,
            repeated_user_features,
            negative_item_features,
        )
        interactions = dataset.data.interactions[
            dataset.data.interactions[:, 0] == repeated_user_id[0]
        ]
        recommendations = negative_item_id[torch.argsort(model_scores)][
            :number_of_recommendations
        ]

        loss_value = loss(interactions, recommendations)
        loss_value.backward()

        optimizer.step()
        optimizer.zero_grad()

        batch_count += 1
        mean_loss += loss_value.item()

        if scores is not None:
            for i, score in enumerate(scores):
                mean_scores[i] += score(interactions, recommendations).item()

    return (
        mean_loss / batch_count,
        [mean_score / batch_count for mean_score in mean_scores],
    )
