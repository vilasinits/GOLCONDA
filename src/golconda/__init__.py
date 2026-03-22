"""
GOLCONDA
========
Generator of Lensing Convergence with Optimized Non-Gaussian
Distribution Adjustment.

Generates weak-lensing convergence (κ) maps that preserve non-Gaussian
statistics by iteratively adjusting wavelet coefficients to match target
power spectra and ℓ₁-norm statistics.

Public API
----------
WaveletDecomposer
    Multi-scale top-hat or starlet wavelet decomposition (JAX-accelerated).
PowerSpectrum
    Flat-sky angular power spectrum computation and Cls adjustment.
SimulationDataLoader
    Load convergence maps from MassiveNuS, SLICS, or HOWLS simulations.
process_image
    Extract per-scale wavelet statistics from a target map.
adjust_map_l1
    Match wavelet coefficient histograms to target ℓ₁-norm statistics.
get_moments_from_map
    Compute mean, variance, skewness, and kurtosis of a map.
"""

from golconda._version import __version__

from golconda.decomposer import WaveletDecomposer
from golconda.adjustcls import PowerSpectrum, PowerSpectrumAdjuster
from golconda.adjustl1 import (
    calculate_histogram_l1norm,
    adjust_map_l1,
    adjust_map_l1_jax,
    process_image,
    adjust_pixel_values,
)
from golconda.loadsimulation import SimulationDataLoader
from golconda.moments import get_moments_from_map
from golconda.image import (
    split_image_into_patches,
    wiener_deconvolution_flat,
    compute_binedges,
)

__all__ = [
    # core pipeline
    "WaveletDecomposer",
    "PowerSpectrum",
    "PowerSpectrumAdjuster",
    "SimulationDataLoader",
    # l1 / histogram utilities
    "calculate_histogram_l1norm",
    "adjust_map_l1",
    "adjust_map_l1_jax",
    "process_image",
    "adjust_pixel_values",
    # validation / helpers
    "get_moments_from_map",
    "split_image_into_patches",
    "wiener_deconvolution_flat",
    "compute_binedges",
    # metadata
    "__version__",
]
