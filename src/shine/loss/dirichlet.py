# written by Dexiang Zong
# CardioVascular Systems Imaging and Artificial Intelligence (CVS.AI)
# National Heart Research Institute Singapore (NHRIS)
# national heart center singarpore

import torch
import torch.nn as nn


class DirichletEnergyLoss(nn.Module):
    """Computes the Dirichlet energy of a neural field, penalizing high frequency spatial oscillations."""
    def __init__(self):
        super().__init__()

    def forward(self, coords: torch.Tensor, pred: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """Computes the norm of gradient of pred with respect to coords.
        
        Args:
            coords: Coordinates of shape (B, 3). Requires grad must be True.
            pred: Neural field prediction of shape (B, 1) or (B,).
            reduction: Reduction type ('mean' or 'none').
            
        Returns:
            dirichlet_energy: Tensor indicating gradient norm penalty.
        """
        gradients = torch.autograd.grad(
            outputs=pred,
            inputs=coords,
            grad_outputs=torch.ones_like(pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True
        )[0]
        
        if gradients is None:
            if reduction == 'mean':
                return torch.tensor(0.0, requires_grad=True, device=pred.device)
            else:
                return torch.zeros(pred.shape[0], requires_grad=True, device=pred.device)
            
        # Norm squared of gradients: ||grad f||^2
        grad_norm_sq = torch.sum(gradients ** 2, dim=-1)
        
        if reduction == 'mean':
            return grad_norm_sq.mean()
        elif reduction == 'none':
            return grad_norm_sq
        else:
            raise ValueError(f"Unknown reduction: {reduction}")


class EntropyWeightedLoss(nn.Module):
    """Combines Binary Cross Entropy Loss with Entropy-Weighted Dirichlet Energy Regularization."""
    def __init__(self, dirichlet_weight: float = 0.01, uncertainty_threshold: float = 0.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dirichlet = DirichletEnergyLoss()
        self.dirichlet_weight = dirichlet_weight
        self.uncertainty_threshold = uncertainty_threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                coords: torch.Tensor, uncertainty: torch.Tensor) -> dict[str, torch.Tensor]:
        """Calculates loss adaptively weighting Dirichlet Energy based on epistemic uncertainty.
        
        Args:
            pred: Predicted scar intensity (logits).
            target: Ground truth scar intensity (only available for observed regions).
            coords: Coordinate tensor where prediction is evaluated.
            uncertainty: Measured epistemic uncertainty (normalized entropy) of the network.
            
        Returns:
            loss_dict: Dictionary containing total loss, BCE loss, and regularizer loss.
        """
        # Data loss on observed regions (where target isn't NaN or mask is applied)
        data_loss = self.bce(pred, target)
        
        # Apply Sigmoid to predicted logits to get probabilities before computing Dirichlet energy
        pred_prob = torch.sigmoid(pred)
        
        # Compute pointwise Dirichlet Energy Loss
        raw_dirichlet = self.dirichlet(coords, pred_prob, reduction='none')
        
        # Weighted Dirichlet energy: higher uncertainty -> enforce smoother predictions in blind zones
        weight = uncertainty
        if self.uncertainty_threshold > 0:
            weight = torch.where(uncertainty > self.uncertainty_threshold, uncertainty, torch.zeros_like(uncertainty))
            
        if weight.dim() > 1:
            weight = weight.squeeze(-1)
            
        weighted_dirichlet = (weight * raw_dirichlet).mean()
        
        total_loss = data_loss + self.dirichlet_weight * weighted_dirichlet
        
        return {
            "loss": total_loss,
            "data_loss": data_loss,
            "dirichlet_loss": weighted_dirichlet
        }
