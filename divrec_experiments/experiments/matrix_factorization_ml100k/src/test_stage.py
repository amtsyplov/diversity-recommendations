import torch
from torch.utils.data import DataLoader

from divrec.loaders import BPRSampling
from divrec.metrics import (
    AUCScore,
    EntropyDiversityScore,
    RecallAtKScore,
    MeanAveragePrecisionAtKScore,
    PrecisionAtKScore,
    NDCGScore
)
from divrec.models import MatrixFactorization
from divrec.utils import get_logger
from divrec_experiments.datasets import MovieLens100K
from divrec_experiments.utils import to_json
from divrec_pipelines.pipeline import Container, stage


def evaluate_auc_roc(
        loader: DataLoader,
        model: MatrixFactorization,
) -> float:
    score_function = AUCScore()
    score_value = 0.0
    batch_count = 0

    for user, positive, negative in loader:
        positive_predictions = model(user, positive)
        negative_predictions = model(user, negative)

        score = score_function(positive_predictions, negative_predictions)
        score_value += score.item()
        batch_count += 1

    return score_value / batch_count


@stage(configuration={
    "model_path": "workdir",
    "embedding_dim": 300,
    "max_sampled": -1,
    "test_scores_filepath": "scores.json",
    "logfile": "logfile.log",
    "k": 10,
})
def test_model(config, arg):
    dataset: MovieLens100K = arg["data"]
    logger = get_logger(__name__, config["logfile"])

    model = MatrixFactorization(dataset.no_users, dataset.no_items, embedding_dim=config["embedding_dim"])
    model.load_state_dict(torch.load(config["model_path"]))
    model.to("cpu")
    model.eval()

    test_dataset = BPRSampling(
        dataset.test,
        user_item_interactions_frozen=torch.concatenate((dataset.train, dataset.validation), dim=0).long(),
        max_sampled=config["max_sampled"],
    )

    test_loader = DataLoader(test_dataset, batch_size=1000)

    scores = {}

    try:
        test_auc_value = evaluate_auc_roc(test_loader, model)
        scores["auc_score"] = test_auc_value
        logger.info(f"Successfully evaluate AUC score: {test_auc_value:.6f}")
    except Exception as exception:
        logger.warn(f"Error while AUC score evaluating:\n{exception}")

    predictions = model.predict_top_k(config["k"])
    logger.info(f"Successfully evaluate model predictions")

    score_function = EntropyDiversityScore()
    try:
        score_value = score_function(dataset.test, predictions)
        scores["entropy_diversity_score"] = score_value.item()
        logger.info(f"Successfully evaluate {type(score_function).__name__} score: {score_value:.6f}")
    except Exception as exception:
        logger.warn(f"Error while {type(score_function).__name__} score evaluating:\n{exception}")

    score_function = RecallAtKScore()
    try:
        score_value = score_function(dataset.test, predictions)
        scores[f"recall@{config['k']}"] = score_value.item()
        logger.info(f"Successfully evaluate {type(score_function).__name__} score: {score_value:.6f}")
    except Exception as exception:
        logger.warn(f"Error while {type(score_function).__name__} score evaluating:\n{exception}")

    score_function = PrecisionAtKScore()
    try:
        score_value = score_function(dataset.test, predictions)
        scores[f"precision@{config['k']}"] = score_value.item()
        logger.info(f"Successfully evaluate {type(score_function).__name__} score: {score_value:.6f}")
    except Exception as exception:
        logger.warn(f"Error while {type(score_function).__name__} score evaluating:\n{exception}")

    score_function = MeanAveragePrecisionAtKScore()
    try:
        score_value = score_function(dataset.test, predictions)
        scores[f"MAP@{config['k']}"] = score_value.item()
        logger.info(f"Successfully evaluate {type(score_function).__name__} score: {score_value:.6f}")
    except Exception as exception:
        logger.warn(f"Error while {type(score_function).__name__} score evaluating:\n{exception}")

    score_function = NDCGScore()
    try:
        score_value = score_function(dataset.test, predictions)
        scores[f"nDCG@{config['k']}"] = score_value.item()
        logger.info(f"Successfully evaluate {type(score_function).__name__} score: {score_value:.6f}")
    except Exception as exception:
        logger.warn(f"Error while {type(score_function).__name__} score evaluating:\n{exception}")

    to_json(scores, config["test_scores_filepath"])
    return Container(elements=scores)
