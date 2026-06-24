# SHINE: Entropy-Guided Digital Twins for Myocardial Scar Reconstruction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-MICCAI%202026-blue)](#)

This repository contains the official implementation of **SHINE** (Scar Heterogeneity via Implicit Neural Encoding), a physics-informed digital twin framework designed to reconstruct high-fidelity, continuous 3D myocardial scars (specifically narrow conducting isthmuses for Ventricular Tachycardia (VT) ablation) from sparse, highly anisotropic clinical LGE-MRI.

---

## 📂 Repository Structure

```text
SHINE_share/
├── configs/
│   └── default.yaml          # Model hyperparameters and optimizer configurations
├── data/                     # Raw patient VTK datasets
│   ├── model.vtk             # 3D heart anatomical ventricular mesh model
│   └── slice1.vtk            # 2D LGE-MRI slice
├── doc/
│   └── Paper-xxxx.pdf        # Accompanying MICCAI 2026 methodology paper
├── scripts/
│   └── Intersect_3D_mask.py  # Script to intersect 2D slices with 3D model
├── src/
│   └── shine/                # Core SHINE package
│       ├── __init__.py
│       ├── dataset.py        # PyTorch dataset for coordinate representation
│       ├── train.py          # Patient-specific manifold fitting execution
│       ├── infer.py          # Continuous scar inference and VTK exporting
│       ├── models/
│       │   └── siren.py      # Bayesian SIREN backbone with MC Dropout
│       ├── loss/
│       │   └── dirichlet.py  # Entropy-Weighted Dirichlet Energy loss
│       └── utils/
│           ├── mesh.py       # Mesh parsing and point sampling
│           └── registration.py
├── tests/                    # Unit testing suite
│   ├── test_loss.py
│   └── test_model.py
├── requirements.txt          # Package dependencies
└── setup.py                  # Package installation script
```

---

## ⚙️ Installation

1. Create and activate a Python virtual environment (Python $\ge 3.8$):
   ```bash
   conda create -n shine python=3.9
   conda activate shine
   ```

2. Install dependencies and the `shine` package in editable mode:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

---

## 🏃 Quick Start Guide

### Step 1: Preprocess and Intersect Data
Run the intersection utility script. It reads the raw VTK files from `data`, checks for mesh-plane intersections, and extracts coordinate-intensity pairs:
```bash
python scripts/Intersect_3D_mask.py
```
This generates `Intersect_data.npy` inside `data/`.

### Step 2: Patient-Specific Manifold Fitting (Training)
Train the Bayesian SIREN model on the intersected data coordinates. The script automatically detects and utilizes GPU acceleration (**MPS** on Apple Silicon or **CUDA** on NVIDIA GPUs):
```bash
python src/shine/train.py \
    --data-path data/Intersect_data.npy \
    --save-path checkpoints/shine_model.pth
```
This saves the trained weights and coordinate scaling metadata to `checkpoints/shine_model.pth`.

### Step 3: Continuous Scar Reconstruction (Inference)
Interpolate the scar field continuously onto the 3D heart anatomical mesh (e.g. `model.vtk` vertices) and export the reconstructed digital twin:
```bash
python src/shine/infer.py \
    --model-path checkpoints/shine_model.pth \
    --mesh-path data/model.vtk \
    --output-path data/model_reconstructed.vtk
```
This generates `data/model_reconstructed.vtk` containing two new point data arrays:
- **`ScarProbability`**: Reconstructed continuous scar intensity/probability.
- **`EpistemicUncertainty`**: Model prediction uncertainty (via MC Dropout normalized entropy passes), highlighting "blind zone" gaps between slices.

---

## 🧪 Running Unit Tests
To verify the installation and core component calculations, run `pytest`:
```bash
pytest
```
All unit tests should pass.

---

## ✍️ Citation
If you find this codebase or our paper useful for your research, please cite:

```bibtex
@inproceedings{zong2026shine,
  title={SHINE: An Entropy-Guided Digital Twin Framework for 3D Myocardial Scar Reconstruction from Sparse CMR},
  author={Zong, Dexiang and others},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year={2026}
}
```
