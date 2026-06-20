# SHINE: Entropy-Guided Digital Twins for Myocardial Scar Reconstruction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-MICCAI%202026-blue)](#)

This repository contains the official implementation of the paper **"SHINE: An Entropy-Guided Digital Twin Framework for 3D Myocardial Scar Reconstruction from Sparse CMR"** [1].

## 🌟 Overview
**SHINE** (Scar Heterogeneity via Implicit Neural Encoding) is a novel physics-informed digital twin framework designed to reconstruct high-fidelity, continuous 3D myocardial scars—specifically the narrow conducting isthmuses essential for Ventricular Tachycardia (VT) ablation [1, 2]. 

Clinical Late Gadolinium Enhancement (LGE) MRI typically suffers from sparse, highly anisotropic through-plane resolution (e.g., 1.2 × 1.2 × 8.0 mm), leaving ~80% of the myocardial volume unobserved as "blind zones" [3]. Traditional discrete interpolations (e.g., Log-Odds) create broken "staircase artifacts", while standard Neural Fields suffer from spectral bias and generate non-physical "hallucinations" [3, 4]. 

SHINE reformulates this highly ill-posed inverse problem as a **continuous function learning task on a patient-specific anatomical manifold**, effectively bridging the Nyquist sampling gap [1, 5].

## 🚀 Core Innovations
1. **Dual-Track Manifold Integration:** SHINE seamlessly aligns sparse LGE-MRI (Pathology Track) with dense Cine-derived 3D anatomical meshes (Anatomy Track) through a Spatio-Temporal Co-registration hub. Crucially, this is achieved **without voxel resampling**, strictly blocking upstream geometric error propagation [6, 7].
2. **Bayesian SIREN Backbone:** Employs a 4-layer Bayesian SIREN architecture (256 hidden units, T=50 Monte Carlo passes) [8]. The sinusoidal periodic activations overcome spectral bias to resolve high-frequency geometric structures (sub-voxel channels), while the variational framework quantifies predictive epistemic uncertainty [9].
3. **Entropy-Weighted Dirichlet Energy Loss:** A novel uncertainty-aware regularization mechanism [9, 10]. It leverages predictive entropy to adaptively enforce harmonic smoothness ($\Delta f = 0$) in data-sparse blind gaps, while rigorously preserving high-fidelity topological details in confident, data-rich regions [8, 10].

## 🏆 Key Performances
Evaluated on a multi-center clinical VT cohort of 40 patients (across 1.5T and 3.0T scanners), SHINE achieves state-of-the-art results [11]:
- **Exceptional Accuracy:** Achieves a Dice Similarity Coefficient (DSC) of **0.93** and a near-perfect intensity fidelity (MSE) of **$4.0 \times 10^{-5}$** under a rigorous Leave-One-Slice-Out (LOSO) cross-validation protocol [11, 12].
- **Clinical Safety Maps:** SHINE outputs intuitive uncertainty-aware safety maps. High entropy (red/yellow) alerts clinicians to unobserved inter-slice gaps, while low entropy (blue) correlates with verified LGE findings [13, 14].
- **Real-Time Efficiency:** While patient-specific manifold fitting is a pre-operative process (~7.5 mins), intraoperative inference takes only **0.54 seconds** per patient on an NVIDIA RTX 4090 GPU, enabling real-time clinical deployment [13].
