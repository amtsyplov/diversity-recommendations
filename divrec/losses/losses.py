import torch


class LogSigmoidDifferenceLoss(torch.nn.Module):
    def __init__(self, reduce: bool = True, reduction: str = "mean"):
        assert reduction in ["node", "mean", "sum"]
        torch.nn.Module.__init__(self)
        self.reduce = reduce and reduction != "none"
        self.reduction = reduction

    def forward(self, x, y):
        loss_values = torch.nn.LogSigmoid()(x - y)
        if not self.reduce:
            return loss_values
        elif self.reduction == "mean":
            return torch.mean(loss_values)
        return torch.sum(loss_values)
