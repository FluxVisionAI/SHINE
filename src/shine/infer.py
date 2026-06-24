# written by Dexiang Zong
# CardioVascular Systems Imaging and Artificial Intelligence (CVS.AI)
# National Heart Research Institute Singapore (NHRIS)
# national heart center singarpore

import argparse
import os
import yaml
import torch
import pyvista as pv
import numpy as np
from shine.models.siren import BayesianSIREN


def main():
    parser = argparse.ArgumentParser(description="SHINE: Patient-Specific Scar Inference")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained checkpoint file (.pth)")
    parser.add_argument("--mesh-path", type=str, required=True, help="Path to input VTK mesh model file (.vtk)")
    parser.add_argument("--output-path", type=str, default="data/model_reconstructed.vtk", help="Path to save output VTK file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Running inference on device: {device}")

    # Load trained checkpoint
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Checkpoint not found at: {args.model_path}")
        
    print(f"Loading checkpoint from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Instantiate model
    model = BayesianSIREN(
        in_features=config["model"]["in_features"],
        out_features=config["model"]["out_features"],
        hidden_features=config["model"]["hidden_features"],
        num_layers=config["model"]["num_layers"],
        first_omega_0=config["model"]["first_omega_0"],
        hidden_omega_0=config["model"]["hidden_omega_0"],
        dropout_rate=config["model"]["dropout_rate"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load input mesh
    if not os.path.exists(args.mesh_path):
        raise FileNotFoundError(f"VTK mesh not found at: {args.mesh_path}")
        
    print(f"Loading anatomical heart mesh from {args.mesh_path}...")
    mesh = pv.read(args.mesh_path)
    raw_vertices = np.array(mesh.points, dtype=np.float32)
    print(f"Loaded mesh with {len(raw_vertices)} vertices.")

    # Scale vertices using training scaler bounds
    min_coords = checkpoint.get("min_coords")
    max_coords = checkpoint.get("max_coords")
    
    if min_coords is not None and max_coords is not None:
        print("Scaling vertices using saved normalization parameters...")
        min_coords_arr = np.array(min_coords, dtype=np.float32)
        max_coords_arr = np.array(max_coords, dtype=np.float32)
        normalized_vertices = (raw_vertices - min_coords_arr) / (max_coords_arr - min_coords_arr + 1e-8)
    else:
        print("Warning: Normalization parameters not found in checkpoint. Running raw coordinates.")
        normalized_vertices = raw_vertices

    # Run uncertainty inference
    vertices_tensor = torch.tensor(normalized_vertices, dtype=torch.float32).to(device)
    
    print(f"Running uncertainty estimation with {config['model']['mc_passes']} Monte Carlo passes...")
    with torch.no_grad():
        mean_reconstruction, uncertainty = model.forward_mc(
            vertices_tensor, 
            num_passes=config["model"]["mc_passes"]
        )

    mean_reconstruction_np = mean_reconstruction.cpu().numpy().squeeze()
    uncertainty_np = uncertainty.cpu().numpy().squeeze()

    # Save outputs back to VTK
    mesh.point_data["ScarProbability"] = mean_reconstruction_np
    mesh.point_data["EpistemicUncertainty"] = uncertainty_np
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    mesh.save(args.output_path)
    print(f"Successfully exported reconstructed heart mesh with scalar predictions to: {args.output_path}")


if __name__ == "__main__":
    main()

