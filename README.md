# SHINE: Entropy-Guided Digital Twins for Myocardial Scar Reconstruction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-MICCAI%202026-blue)](#)

This repository contains the official implementation of the core reconstruction algorithm from the paper **"SHINE: An Entropy-Guided Digital Twin Framework for 3D Myocardial Scar Reconstruction from Sparse CMR"**.

## 📌 Scope of this Repository
Please note that this codebase focuses strictly on the core methodological contributions of SHINE (the **Entropy-Guided Neural Field** and **Manifold Integration**). 

The upstream preprocessing steps—specifically the dense Cine-MRI anatomical segmentation and sparse LGE-MRI pathology segmentation (scar/myocardium)—are generated using our dedicated end-to-end deep learning pipeline. If you are interested in the segmentation methodology, please refer to our accompanying paper published in the *Journal of Cardiovascular Magnetic Resonance* (JCMR, 2026) detailed below.

**What is included:**
We provide the complete reconstruction pipeline that takes **a pre-extracted 3D anatomical mesh** and **a few sparse LGE probability slices** as inputs, and reconstructs the high-fidelity continuous 3D digital twin. A sample dataset (1 patient mesh + corresponding LGE slices) is provided for out-of-the-box testing.

## 🔗 Related Work: Upstream Segmentation Pipeline
For the upstream LGE scar and anatomical segmentation pipeline utilized to generate the inputs for SHINE, please refer to our related work:

> **Prognostic value of end-to-end deep learning assessment of myocardial scar and microvascular obstruction on late gadolinium enhancement cardiovascular magnetic resonance.**
> *Pei Yang, Shuang Leng, Dexiang Zong, et al.*
> Journal of Cardiovascular Magnetic Resonance, 28(1), 102712, 2026.
> [DOI: 10.1016/j.jocmr.2026.102712](https://doi.org/10.1016/j.jocmr.2026.102712)

## 🌟 Overview
**SHINE** (Scar Heterogeneity via Implicit Neural Encoding) is a novel physics-informed digital twin framework designed to reconstruct high-fidelity, continuous 3D myocardial scars—specifically the narrow conducting isthmuses essential for Ventricular Tachycardia (VT) ablation. 

Clinical Late Gadolinium Enhancement (LGE) MRI typically suffers from sparse, highly anisotropic through-plane resolution (e.g., 1.2 × 1.2 × 8.0 mm), leaving ~80% of the myocardial volume unobserved as "blind zones". SHINE reformulates this highly ill-posed inverse problem as a **continuous function learning task on a patient-specific anatomical manifold**, effectively bridging the Nyquist sampling gap without relying on discrete "staircase" interpolations.

## 🚀 Core Innovations Implemented
1. **Spatio-Temporal Co-registration (No Voxel Resampling):** The code smoothly anchors sparse LGE spatial slices onto the continuous 3D anatomical mesh, strictly blocking upstream geometric error propagation.
2. **Bayesian SIREN Backbone:** Employs a 4-layer Bayesian SIREN architecture (256 hidden units, T=50 Monte Carlo passes). The sinusoidal periodic activations overcome spectral bias to resolve high-frequency geometric structures (sub-voxel channels).
3. **Entropy-Weighted Dirichlet Energy Loss:** A novel uncertainty-aware regularization mechanism. It leverages predictive epistemic uncertainty to adaptively enforce harmonic smoothness ($\Delta f = 0$) in data-sparse blind gaps, while rigorously preserving high-fidelity topological details in confident, data-rich regions.

## 🏆 Key Clinical Performances (from the paper)
Evaluated on a multi-center clinical VT cohort, SHINE achieves state-of-the-art results:
- **Exceptional Accuracy:** Achieves a Dice Similarity Coefficient (DSC) of **0.93** and a near-perfect intensity fidelity (MSE) of **$4.0 \times 10^{-5}$** under a rigorous Leave-One-Slice-Out (LOSO) cross-validation protocol.
- **Real-Time Efficiency:** While patient-specific manifold fitting is a pre-operative process (~7.5 mins), intraoperative inference takes only **0.54 seconds** per patient on an NVIDIA RTX 4090 GPU.
