import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stablemodels.lyapunov import LyapunovFunction

class StableDynamicsModel(nn.Module):
    def __init__(self,
                 input_shape,
                 alpha=0.9,
                 layer_sizes=[64, 64],
                 lr=3e-4,
                 lyapunov_lr=3e-4,
                 lyapunov_eps=1e-3):
        super(StableDynamicsModel, self).__init__()
        input_shape = np.prod(input_shape)
        self._layer_sizes = layer_sizes
        self._lr = lr
        self._alpha = alpha
        self._lyapunov_function = LyapunovFunction((input_shape,),
                                                   layer_sizes=layer_sizes + [1],
                                                   lr=lyapunov_lr,
                                                   eps=lyapunov_eps)
        self._l1 = nn.Linear(input_shape, 64)
        self._l2 = nn.Linear(64, 64)
        self._l3 = nn.Linear(64, input_shape)

    def forward(self, x):
        x.retain_grad()
        f = self._l1(x.squeeze()).relu()
        f = self._l2(f).relu()
        f = self._l3(f)
        if len(f.shape) == 1:
            f.unsqueeze_(0)
        lyapunov = self._lyapunov_function(f).squeeze()
        lyapunov.backward(gradient=torch.ones_like(lyapunov), retain_graph=True)
        grad_v = x.grad
        gv = grad_v.view(-1, 1, *f.shape[1:])
        fv = grad_v.view(-1, *f.shape[1:], 1)
        dot = (gv @ fv).squeeze()
        orth = (dot + self._alpha * lyapunov).squeeze().relu() / grad_v.pow(2).sum()
        if len(orth.shape) == 0:
            orth.unsqueeze_(0)
        return f - torch.diag_embed(orth) @ grad_v
