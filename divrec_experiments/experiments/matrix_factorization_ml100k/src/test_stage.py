import torch
from torch.utils.data import DataLoader

from divrec.datasets import BPRSampling
from divrec.metrics import AUCScore, EntropyDiversityScore
from divrec.models import MatrixFactorization
from divrec_experiments.datasets import MovieLens100K
from divrec_experiments.pipeline import stage


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
    "model_path": "models",
    "embedding_dim": 300,
    "max_sampled": -1,
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

    test_auc_score = evaluate_model(test_loader, model, score_function)
    print(f"Test AUC: {test_auc_score:.6f}")

    predictions = model.predict_top_k(10)
    entropy_diversity_score = EntropyDiversityScore()
    entropy_diversity_value = entropy_diversity_score(predictions).item()
    print(f"Diversity by entropy: {entropy_diversity_value:.6f}")
