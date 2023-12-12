from typing import Tuple

import torch

from divrec.datasets.base_datasets import UserItemInteractionsDataset


def train_validation_split(
    dataset: UserItemInteractionsDataset, validation_size: int
) -> Tuple[UserItemInteractionsDataset, UserItemInteractionsDataset]:
    """
    Divides UserItemInteractionsDataset interactions into two parts:
    train and validation. For each user expects at least `validation_size`
    interactions counts. Extract last (by sorting) `validation_size`
    interactions for each user into validation sample and train sample
    is all other interactions.

    item_features
user_features
number_of_items
number_of_users

    :param dataset: train data
    :param validation_size: number of items per user used for validation
    :return: tuple of train and validation dataset
    """
    train_interactions = list()
    train_interactions_scores = list()

    validation_interactions = list()
    validation_interactions_scores = list()

    for user_id in range(dataset.number_of_users):
        binary_mask = dataset.interactions[:, 0] == user_id
        user_interactions = dataset.interactions[binary_mask]
        user_interactions_scores = dataset.interaction_scores[binary_mask]

        train_interactions.append(user_interactions[:-validation_size])
        train_interactions_scores.append(user_interactions_scores[:-validation_size])

        validation_interactions.append(user_interactions[-validation_size:])
        validation_interactions_scores.append(
            user_interactions_scores[-validation_size:]
        )

    train = UserItemInteractionsDataset(
        interactions=torch.concatenate(train_interactions).long(),
        interaction_scores=torch.concatenate(train_interactions_scores),
        item_features=dataset.item_features,
        user_features=dataset.user_features,
        number_of_items=dataset.number_of_items,
        number_of_users=dataset.number_of_users,
    )

    validation = UserItemInteractionsDataset(
        interactions=torch.concatenate(validation_interactions).long(),
        interaction_scores=torch.concatenate(validation_interactions_scores),
        item_features=dataset.item_features,
        user_features=dataset.user_features,
        number_of_items=dataset.number_of_items,
        number_of_users=dataset.number_of_users,
    )

    return train, validation
