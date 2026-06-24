# written by Dexiang Zong
# CardioVascular Systems Imaging and Artificial Intelligence (CVS.AI)
# National Heart Research Institute Singapore (NHRIS)
# national heart center singarpore

import math
import torch
import torch.nn as nn


class Sine(nn.Module):
    """Sine activation function with scaling factor w0."""
    def __init__(self, w0: float = 30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class SIRENLayer(nn.Module):
    """A single layer of a SIREN model."""
    def __init__(self, in_features: int, out_features: int, is_first: bool = False,
                 shared_w0: float = 30.0, dropout_rate: float = 0.0):
        super().__init__()
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.activation = Sine(shared_w0)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None
        self.init_weights(in_features, shared_w0)

    def init_weights(self, in_features: int, w0: float):
        """Initializes weights using the custom SIREN schema."""
        with torch.no_grad():
            if self.is_first:
                bound = 1 / in_features
                self.linear.weight.uniform_(-bound, bound)
            else:
                bound = math.sqrt(6 / in_features) / w0
                self.linear.weight.uniform_(-bound, bound)
            # Initialize bias to zero
            if self.linear.bias is not None:
                self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class BayesianSIREN(nn.Module):
    """Bayesian Sinusoidal Representation Network (SIREN) for coordinate learning.
    
    Uses Dropout at test time to perform Monte Carlo (MC) Dropout for epistemic
    uncertainty estimation.
    """
    def __init__(self, in_features: int = 3, out_features: int = 1,
                 hidden_features: int = 256, num_layers: int = 4,
                 first_omega_0: float = 30.0, hidden_omega_0: float = 30.0,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        layers = []
        # First layer
        layers.append(SIRENLayer(in_features, hidden_features, is_first=True,
                                 shared_w0=first_omega_0, dropout_rate=dropout_rate))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(SIRENLayer(hidden_features, hidden_features, is_first=False,
                                     shared_w0=hidden_omega_0, dropout_rate=dropout_rate))
            
        self.net = nn.Sequential(*layers)
        
        # Final prediction layer
        self.final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            bound = math.sqrt(6 / hidden_features) / hidden_omega_0
            self.final_linear.weight.uniform_(-bound, bound)
            if self.final_linear.bias is not None:
                self.final_linear.bias.zero_()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        x = self.net(coords)
        return self.final_linear(x)

    def forward_mc(self, coords: torch.Tensor, num_passes: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs multiple forward passes with Dropout enabled to compute mean prediction and epistemic uncertainty.
        
        Args:
            coords: Input coordinates of shape (B, in_features).
            num_passes: Number of Monte Carlo iterations.
            
        Returns:
            mean: Average prediction of shape (B, out_features).
            uncertainty: Epistemic uncertainty (e.g. normalized entropy) of shape (B, out_features).
        """
        # Ensure dropout is active even during inference/evaluation
        was_training = self.training
        self.train() 
        
        predictions = []
        for _ in range(num_passes):
            # Apply sigmoid to raw predictions to get probabilities
            predictions.append(torch.sigmoid(self.forward(coords)))
            
        predictions = torch.stack(predictions, dim=0) # (num_passes, B, out_features)
        
        mean = predictions.mean(dim=0)
        
        # Compute normalized binary entropy in base 2 (so max entropy is 1.0)
        mean_clamped = torch.clamp(mean, min=1e-8, max=1.0 - 1e-8)
        entropy = - mean_clamped * torch.log2(mean_clamped) - (1.0 - mean_clamped) * torch.log2(1.0 - mean_clamped)
        
        # Restore training state
        if not was_training:
            self.eval()
            
        return mean, entropy
