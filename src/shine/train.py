# written by Dexiang Zong
# CardioVascular Systems Imaging and Artificial Intelligence (CVS.AI)
# National Heart Research Institute Singapore (NHRIS)
# national heart center singarpore

import argparse
import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from shine.models.siren import BayesianSIREN
from shine.loss.dirichlet import EntropyWeightedLoss
from shine.dataset import ManifoldDataset


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="SHINE: Patient-Specific Manifold Fitting")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--data-path", type=str, default=None, help="Path to Intersect_data.npy file")
    parser.add_argument("--save-path", type=str, default="checkpoints/shine_model.pth", help="Path to save trained checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Training on device: {device}")

    epochs = args.epochs if args.epochs is not None else config["training"]["epochs"]

    min_coords = None
    max_coords = None

    if args.data_path is not None and os.path.exists(args.data_path):
        print(f"Loading real data from {args.data_path}...")
        data = np.load(args.data_path)
        
        # Filter out background or invalid points (where scar intensity is -1.0)
        data = data[data[:, 3] >= 0.0]
        
        raw_coords = data[:, 0:3]
        raw_intensities = data[:, 3:4]
        
        # Normalize coordinates to [0, 1]
        min_coords = raw_coords.min(axis=0)
        max_coords = raw_coords.max(axis=0)
        normalized_coords = (raw_coords - min_coords) / (max_coords - min_coords + 1e-8)
        
        coords_tensor = torch.tensor(normalized_coords, dtype=torch.float32)
        intensities_tensor = torch.tensor(raw_intensities, dtype=torch.float32)
        print(f"Loaded {len(coords_tensor)} data points after filtering.")
    else:
        print("Using dummy dataset for skeleton initialization...")
        # Dummy dataset for skeleton initialization
        dummy_coords = torch.randn(10000, 3)
        dummy_intensities = torch.rand(10000, 1)
        coords_tensor = dummy_coords
        intensities_tensor = dummy_intensities

    dataset = ManifoldDataset(coords_tensor, intensities_tensor)
    dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)

    # Initialize model
    model = BayesianSIREN(
        in_features=config["model"]["in_features"],
        out_features=config["model"]["out_features"],
        hidden_features=config["model"]["hidden_features"],
        num_layers=config["model"]["num_layers"],
        first_omega_0=config["model"]["first_omega_0"],
        hidden_omega_0=config["model"]["hidden_omega_0"],
        dropout_rate=config["model"]["dropout_rate"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = EntropyWeightedLoss(
        dirichlet_weight=config["loss"]["dirichlet_weight"],
        uncertainty_threshold=config["loss"].get("uncertainty_threshold", 0.0)
    )

    print("Starting patient-specific manifold fitting...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for coords, target in dataloader:
            coords = coords.to(device).requires_grad_(True)
            target = target.to(device)
            
            optimizer.zero_grad()
            
            # Predict
            pred = model(coords)
            
            # Compute epistemic uncertainty using Bayesian Dropout passes
            with torch.no_grad():
                _, uncertainty = model.forward_mc(coords, num_passes=5) # Reduced passes during train for speed
                
            loss_dict = criterion(pred, target, coords, uncertainty)
            loss = loss_dict["loss"]
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        if epoch % config["training"]["log_interval"] == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss/len(dataloader):.6f}")

    print("Optimization completed!")
    
    # Save checkpoint
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "min_coords": min_coords.tolist() if min_coords is not None else None,
        "max_coords": max_coords.tolist() if max_coords is not None else None,
        "config": config
    }
    torch.save(checkpoint, args.save_path)
    print(f"Saved trained checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()

