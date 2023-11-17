import torch


class AUCScore(torch.nn.Module):
    def __init__(self, reduce: bool = True, reduction: str = "mean"):
        assert reduction in ["node", "mean", "sum"]
        torch.nn.Module.__init__(self)
        self.reduce = reduce and reduction != "none"
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        loss_values = (x >= y).long()
        if not self.reduce:
            return loss_values
        elif self.reduction == "mean":
            return torch.mean(loss_values)
        return torch.sum(loss_values)
