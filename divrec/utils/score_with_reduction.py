import torch


class ScoreWithReduction(torch.nn.Module):
    def __init__(self, reduce: bool = True, reduction: str = "mean"):
        assert reduction in ["node", "mean", "sum"]
        torch.nn.Module.__init__(self)
        self.reduce = reduce and reduction != "none"
        self.reduction = reduction

    def reduce_loss_values(self, loss_values: torch.Tensor):
        if not self.reduce:
            return loss_values
        elif self.reduction == "mean":
            return torch.sum(loss_values, dim=0) / loss_values.size(0)
        return torch.sum(loss_values, dim=0)
