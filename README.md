# Stable Dynamics Models
![](https://github.com/harwiltz/stable-dynamics-models/workflows/Python%203.7%20Test/badge.svg)

This repo contains an implementation of the method introduced in ["Learning Stable Deep Dynamics
Models"](https://papers.nips.cc/paper/9292-learning-stable-deep-dynamics-models) with PyTorch. This
includes a PyTorch implementation of an [Input Convex Neural
Network](http://proceedings.mlr.press/v70/amos17b/amos17b.pdf).

The goal with this repo is to extend the results to account for controlled systems.

To get started with this library, install like so:

```bash
git clone https://github.com/harwiltz/stable-dynamics-models.git
cd stable-dynamics-models
pip install -r --user requirements.txt
python setup.py install --user
```

You can then instantiate an instance of a stable dynamics model with

```python
from stablemodels import StableDynamicsModel
model = StableDynamicsModel((4,),                 # input shape
                            control_size=2,       # action size
                            alpha=0.9,            # lyapunov constant
                            layer_sizes=[64, 64], # NN layer sizes for lyapunov
                            lr=3e-4,              # learning rate for dynamics model
                            lyapunov_lr=3e-4,     # learning rate for lyapunov function
                            lyapunov_eps=1e-3)    # penalty for equilibrium away from 0

prediction = model(state, action)
```
