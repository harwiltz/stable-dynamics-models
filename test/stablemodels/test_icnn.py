import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from stablemodels.icnn import ICNN

class TestICNN(unittest.TestCase):
    def test_icnn_pipeline_nonbatch(self):
        net = ICNN((4,), [64, 64, 1], F.relu)
        x = torch.ones((4, )).unsqueeze(0)
        y = net(x)

    def test_icnn_pipeline_batch(self):
        net = ICNN((4,), [64, 64, 1], F.relu)
        x = torch.ones((32, 4))
        y = net(x)

    def test_icnn_pipeline_scalar_nonbatch(self):
        net = ICNN((1,), [64, 64, 1], F.relu)
        x = torch.ones((1, )).unsqueeze(0)
        y = net(x)

    def test_icnn_pipeline_scalar_batch(self):
        net = ICNN((1,), [64, 64, 1], F.relu)
        x = torch.ones((1000,))
        y = net(x)

if __name__ == "__main__":
    unittest.main()
