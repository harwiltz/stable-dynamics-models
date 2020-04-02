import unittest

import torch

from stablemodels.model import StableDynamicsModel

class TestStableDynamicsModel(unittest.TestCase):
    def test_stable_model_pipeline(self):
        net = StableDynamicsModel((4,))
        x = torch.randn(4, requires_grad=True).unsqueeze(0)
        next_x = net(x)
        x_shape = x.squeeze().shape
        next_x_shape = next_x.squeeze().shape
        assert x_shape == next_x_shape, "{} =/= {}".format(x_shape, next_x_shape)

    def test_stable_model_batch_pipeline(self):
        net = StableDynamicsModel((4,))
        x = torch.randn((2,4), requires_grad=True).unsqueeze(0)
        next_x = net(x)
        x_shape = x.squeeze().shape
        next_x_shape = next_x.squeeze().shape
        assert x_shape == next_x_shape, "{} =/= {}".format(x_shape, next_x_shape)

    def test_stable_model_backprop(self):
        net = StableDynamicsModel((4,))
        x = torch.randn(4, requires_grad=True).unsqueeze(0)
        next_x = net(x).squeeze()
        next_x.sum().backward()

if __name__ == "__main__":
#    TestStableDynamicsModel().test_stable_model_pipeline()
    unittest.main()
