# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GOLCONDA is a Python library for generating weak lensing convergence (κ) maps that preserve non-Gaussian statistics. It emulates expensive N-body simulation outputs by iteratively adjusting wavelet coefficients to match target power spectra and ℓ1-norm statistics.

## Commands

**Install:**
```bash
pip install .
```

**Build documentation (from `docs/` directory):**
```bash
make html
```

**Regenerate API docs from source:**
```bash
sphinx-apidoc -Mfeo docs src/golconda
```

There are no automated tests in this repository.

## Architecture

The library implements an iterative emulation pipeline:

1. **Load** — `src/loadsimulation.py`: `SimulationDataLoader` reads convergence maps from MassiveNuS (FITS), SLICS (binary), or HOWLS (FITS) formats, auto-detecting type from filename.

2. **Decompose** — `src/decomposer.py`: `WaveletDecomposer` performs multi-scale starlet wavelet decomposition using JAX.

3. **Adjust power spectrum** — `src/adjustcls.py`: `PowerSpectrumAdjuster` uses pyFFTW to compute 1D power spectra from 2D maps and scales Fourier coefficients to match a target spectrum.

4. **Adjust L1-norm** — `src/adjustl1.py`: Adjusts marginal distributions of wavelet coefficients per scale to match target ℓ1-norm statistics, enforcing boundary constraints.

5. **Validate** — `src/moments.py`: Computes mean, variance, skewness, kurtosis for map validation.

6. **Utilities** — `src/image.py`: Patch splitting, Wiener deconvolution, and other image-processing helpers.

The `src/golconda/` subdirectory mirrors the `src/` layout and is the installable package. Primary usage is through Jupyter notebooks in `notebooks/`; there is no CLI.

## Tech Stack

- **JAX** — wavelet decomposition (GPU-acceleratable)
- **pyFFTW** — fast Fourier transforms for power spectrum operations
- **pyccl / lenspack** — cosmology utilities
- **astropy / healpy** — FITS I/O and HEALPix support
- **Sphinx + numpydoc** — documentation (NumPy-style docstrings expected)
- Python ≥ 3.12
