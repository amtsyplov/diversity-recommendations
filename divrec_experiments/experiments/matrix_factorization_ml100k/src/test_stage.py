import json

import torch
from torch.utils.data import DataLoader

from divrec.datasets import BPRSampling
from divrec.metrics import AUCScore, EntropyDiversityScore
from divrec.metrics import recall_at_k, average_precision_at_k, precision_at_k, normalized_discounted_cumulative_gain
from divrec.models import MatrixFactorization
from divrec.utils import get_logger
from divrec_experiments.datasets import MovieLens100K
from divrec_experiments.utils import to_json
from divrec_pipelines.pipeline import Container, stage


def evaluate_default_recommendation_metrics(interactions: torch.LongTensor, recommendations: torch.LongTensor):
    k = recommendations.size(1)
    return {
        f"recall@{k}": torch.mean(recall_at_k(interactions, recommendations)).item(),
        f"precision@{k}": torch.mean(precision_at_k(interactions, recommendations)).item(),
        f"MAP@{k}": torch.mean(average_precision_at_k(interactions, recommendations)).item(),
        f"nDCG@{k}": torch.mean(normalized_discounted_cumulative_gain(interactions, recommendations)).item(),
    }


def evaluate_model(
        loader: DataLoader,
        model: MatrixFactorization,
        score_function: torch.nn.Module
) -> float:
    score_value = 0.0
    batch_count = 0

    model.eval()
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

    model = MatrixFactorization(dataset.no_users, dataset.no_items, embedding_dim=config["embedding_dim"])
    model.to("cpu")

    test_dataset = BPRSampling(
        dataset.test,
        user_item_interactions_frozen=torch.concatenate((dataset.train, dataset.validation), dim=0).long(),
        max_sampled=config["max_sampled"],
    )

    test_loader = DataLoader(test_dataset, batch_size=1000)
    score_function = AUCScore()

    model.load_state_dict(torch.load(config["model_path"]))

    test_auc_value = evaluate_model(test_loader, model, score_function)

    predictions = model.predict_top_k(config["k"])
    entropy_diversity_score = EntropyDiversityScore()
    entropy_diversity_value = entropy_diversity_score(predictions).item()

    scores = {
        "auc_score": test_auc_value,
        "entropy_diversity_score": entropy_diversity_value,
    }
    scores.update(evaluate_default_recommendation_metrics(dataset.test, predictions))

    logger = get_logger(__name__, config["logfile"])
    logger.info("Test scores:\n" + "\n".join([f"{k}: {v:.6f}" for k, v in scores.items()]))

    to_json(scores, config["test_scores_filepath"])
    return Container(elements=scores)
