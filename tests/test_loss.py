# written by Dexiang Zong
# CardioVascular Systems Imaging and Artificial Intelligence (CVS.AI)
# National Heart Research Institute Singapore (NHRIS)
# national heart center singarpore

import torch
from shine.loss.dirichlet import DirichletEnergyLoss, EntropyWeightedLoss


def test_dirichlet_energy_computation():
    coords = torch.randn(5, 3, requires_grad=True)
    # Simple linear function: f(x, y, z) = x + 2y + 3z
    # Norm of grad: sqrt(1^2 + 2^2 + 3^2) = sqrt(14)
    # Norm squared of grad: 14
    pred = coords[:, 0:1] + 2.0 * coords[:, 1:2] + 3.0 * coords[:, 2:3]
    
    dirichlet = DirichletEnergyLoss()
    
    # Test reduction='mean'
    energy_mean = dirichlet(coords, pred, reduction='mean')
    assert torch.allclose(energy_mean, torch.tensor(14.0))
    
    # Test reduction='none'
    energy_none = dirichlet(coords, pred, reduction='none')
    assert energy_none.shape == (5,)
    assert torch.allclose(energy_none, torch.ones(5) * 14.0)


def test_entropy_weighted_loss():
    pred = torch.tensor([[0.5], [0.8]], requires_grad=True)
    target = torch.tensor([[0.0], [1.0]])
    coords = torch.randn(2, 3, requires_grad=True)
    uncertainty = torch.tensor([[0.1], [0.9]])
    
    criterion = EntropyWeightedLoss(dirichlet_weight=0.01, uncertainty_threshold=0.0)
    loss_dict = criterion(pred, target, coords, uncertainty)
    
    assert "loss" in loss_dict
    assert "data_loss" in loss_dict
    assert "dirichlet_loss" in loss_dict
    assert loss_dict["loss"] > 0
    assert not torch.isnan(loss_dict["loss"])
    
    # Verify BCE data loss calculation
    expected_bce = torch.nn.BCEWithLogitsLoss()(pred, target)
    assert torch.allclose(loss_dict["data_loss"], expected_bce)


def test_entropy_weighted_loss_thresholding():
    pred = torch.tensor([[0.5], [0.8]], requires_grad=True)
    target = torch.tensor([[0.0], [1.0]])
    coords = torch.randn(2, 3, requires_grad=True)
    uncertainty = torch.tensor([[0.2], [0.8]])
    
    # Use uncertainty threshold of 0.5: first sample weight becomes 0.0, second stays 0.8
    criterion = EntropyWeightedLoss(dirichlet_weight=0.01, uncertainty_threshold=0.5)
    loss_dict = criterion(pred, target, coords, uncertainty)
    
    # Let's verify that the weighted dirichlet matches (weight * raw_dirichlet).mean()
    pred_prob = torch.sigmoid(pred)
    raw_dirichlet = DirichletEnergyLoss()(coords, pred_prob, reduction='none')
    expected_weight = torch.tensor([0.0, 0.8], device=pred.device)
    expected_weighted_dirichlet = (expected_weight * raw_dirichlet).mean()
    
    assert torch.allclose(loss_dict["dirichlet_loss"], expected_weighted_dirichlet)
