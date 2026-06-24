# written by Dexiang Zong
# CardioVascular Systems Imaging and Artificial Intelligence (CVS.AI)
# National Heart Research Institute Singapore (NHRIS)
# national heart center singarpore

import torch
from shine.models.siren import BayesianSIREN, SIRENLayer, Sine


def test_sine_activation():
    sine = Sine(w0=30.0)
    x = torch.zeros(10)
    y = sine(x)
    assert torch.allclose(y, torch.zeros(10))


def test_siren_layer_dimensions():
    layer = SIRENLayer(in_features=3, out_features=16)
    x = torch.randn(5, 3)
    y = layer(x)
    assert y.shape == (5, 16)


def test_bayesian_siren_forward_mc():
    model = BayesianSIREN(in_features=3, out_features=1, hidden_features=32, num_layers=3)
    
    # Ensure model is in eval mode initially
    model.eval()
    assert not model.training
    
    coords = torch.randn(10, 3)
    mean, entropy = model.forward_mc(coords, num_passes=5)
    
    # Check shape
    assert mean.shape == (10, 1)
    assert entropy.shape == (10, 1)
    
    # Check values are valid probabilities and normalized entropy in [0, 1]
    assert torch.all(mean >= 0.0) and torch.all(mean <= 1.0)
    assert torch.all(entropy >= 0.0) and torch.all(entropy <= 1.0)
    
    # Check training state is correctly restored to eval
    assert not model.training
    
    # If starting in train, it should restore to train
    model.train()
    assert model.training
    _, _ = model.forward_mc(coords, num_passes=2)
    assert model.training
