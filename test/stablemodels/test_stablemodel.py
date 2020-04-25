import unittest

import torch

from stablemodels.model import StableDynamicsModel

class TestStableDynamicsModel(unittest.TestCase):
    def test_stable_model_pipeline(self):
        net = StableDynamicsModel((4,))
        x = torch.randn(4).unsqueeze(0)
        next_x = net(x)
        x_shape = x.squeeze().shape
        next_x_shape = next_x.squeeze().shape
        assert x_shape == next_x_shape, "{} =/= {}".format(x_shape, next_x_shape)

    def test_stable_model_batch_pipeline(self):
        net = StableDynamicsModel((4,))
        x = torch.randn((2,4)).unsqueeze(0)
        next_x = net(x)
        x_shape = x.squeeze().shape
        next_x_shape = next_x.squeeze().shape
        assert x_shape == next_x_shape, "{} =/= {}".format(x_shape, next_x_shape)

    def test_stable_model_backprop(self):
        net = StableDynamicsModel((4,))
        x = torch.randn(4).unsqueeze(0)
        next_x = net(x).squeeze()
        next_x.sum().backward()

    def test_naive_control_stable_model_pipeline(self):
        net = StableDynamicsModel(4, control_size=2)
        x = torch.randn(4).unsqueeze(0)
        u = torch.randn(2).unsqueeze(0)
        next_x = net(x, u)
        x_shape = x.squeeze().shape
        next_x_shape = next_x.squeeze().shape
        assert x_shape == next_x_shape, "{} =/= {}".format(x_shape, next_x_shape)

    def test_naive_control_stable_model_batch_pipeline(self):
        net = StableDynamicsModel(4, control_size=2)
        x = torch.randn((2,4)).unsqueeze(0)
        u = torch.randn((2,2)).unsqueeze(0)
        next_x = net(x, u)
        x_shape = x.squeeze().shape
        next_x_shape = next_x.squeeze().shape
        assert x_shape == next_x_shape, "{} =/= {}".format(x_shape, next_x_shape)

    def test_naive_control_stable_model_backprop(self):
        net = StableDynamicsModel(4, control_size=2)
        x = torch.randn(4).unsqueeze(0)
        u = torch.randn(2).unsqueeze(0)
        next_x = net(x, u).squeeze()
        next_x.sum().backward()

    def test_multistep_prediction(self):
        S = 4
        A = 2
        H = 5
        B = 10
        net = StableDynamicsModel(S, control_size=A)
        x = torch.randn((B, H, S), requires_grad=True)
        u = torch.randn((B, H, A), requires_grad=True)
        prediction = net.predict(x, u)
        assert prediction.shape == x.shape, "{} =/= {}".format(x.shape, prediction.shape)

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
#    TestStableDynamicsModel().test_naive_control_stable_model_batch_pipeline()
    unittest.main()
