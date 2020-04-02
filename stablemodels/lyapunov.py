import torch
import torch.nn as nn

from stablemodels.icnn import ICNN

class LyapunovFunction(nn.Module):
    def __init__(self,
                 input_shape,
                 smooth_relu_thresh=0.1,
                 layer_sizes=[64, 64],
                 lr=3e-4,
                 eps=1e-3):
        super(LyapunovFunction, self).__init__()
        self._d = smooth_relu_thresh
        self._icnn = ICNN(input_shape, layer_sizes, self.smooth_relu)
        self._eps = eps

    def forward(self, x):
        g = self._icnn(x)
        g0 = self._icnn(torch.zeros_like(x))
        return self.smooth_relu(g - g0) + self._eps * x.pow(2).sum()

    def smooth_relu(self, x):
        relu = x.relu()
        # TODO: Is there a clean way to avoid computing both of these on all elements?
        sq = x.pow(2) / (2 * self._d)
        lin = x - self._d/2

        return torch.where(relu < self._d, sq, lin)
