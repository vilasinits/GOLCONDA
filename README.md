# Overview

**GOLCONDA** (Generative modeling of convergence maps based on predicted one-point statistics) is an efficient emulator for generating weak lensing convergence ($\kappa$) maps that capture **non-Gaussian information** beyond traditional power spectra. It synthesizes maps directly from input cosmological statistics, eliminating reliance on computationally expensive N-body simulations.

## Key Features
- 🎯 **Input Flexibility**: Accepts theoretical predictions or precomputed:
  - Power spectrum (two-point statistics)
  - Wavelet $\ell_1$-norm (higher-order statistics)
- ⚙️ **Core Method**: Iterative optimization of wavelet coefficients to match:
  - Target power spectrum
  - Marginal distributions and inter-scale dependencies
- 📊 **High-Fidelity Output**: Generates $\kappa$ maps preserving:
  - Input power spectra
  - Higher-order statistical properties
  - Non-Gaussian information from nonlinear structure formation

## Advantages
- ⚡ **faster** than traditional simulations
- 💻 **Reduced computational requirements**
- 🔍 **Accurate reproduction** of cosmological statistics
- 🚀 **Scalable** for large-scale survey analyses (LSST, Euclid)

## GitHub Repository
https://github.com/vilasinits/GOLCONDA/tree/refactor

## Use Cases
Cosmological parameter inference with non-Gaussian statistics

Pipeline validation for Stage-IV lensing surveys

Fast generation of training data for machine learning models

Covariance matrix estimation beyond power spectra

## Contact

Please feel free to reach out to **tsvilasini97@gmail.com** in case of any bugs or questions regarding usage.


## How to Install

Clone the repo and run:

```bash
pip install .

