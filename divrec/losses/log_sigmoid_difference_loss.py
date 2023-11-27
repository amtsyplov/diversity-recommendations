import torch
from divrec.utils import ScoreWithReduction


class LogSigmoidDifferenceLoss(ScoreWithReduction):
    def __init__(self, *args, **kwargs):
        ScoreWithReduction.__init__(self, *args, *kwargs)
        self.log_sigmoid = torch.nn.LogSigmoid()

    def forward(self, x, y):
        loss_values = self.log_sigmoid(x - y)
        return self.reduce_loss_values(loss_values)
