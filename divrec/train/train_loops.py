import torch
from torch.utils.data import DataLoader
from typing import Optional, Tuple


def bpr_train_loop(
        loader: DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_function: torch.nn.Module,
        score_function: Optional[torch.nn.Module] = None,
) -> Tuple[float, float]:
    losses = list()
    scores = list()

    model.train()
    for users, positives, negatives in loader:
        optimizer.zero_grad()

        positive_predictions = model(users, positives)
        negative_predictions = model(users, negatives)

        loss = loss_function(positive_predictions, negative_predictions)
        losses.append(loss.item())
        loss.backward()

        optimizer.step()

        if score_function is not None:
            score = score_function(positive_predictions, negative_predictions)
            scores.append(score.item())

    batch_avg_loss = sum(losses) / len(losses)
    batch_avg_score = sum(scores) / len(scores) if len(scores) > 0 else 0.
    return batch_avg_loss, batch_avg_score
