import os
import sys

# Project Root Setup (Avoiding Duplicate Entries)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

print("Global imports loaded successfully!")

# Standard Libraries
import time
from typing import List

# Numerical Libraries
import numpy as np
import jax.numpy as jnp
import scipy
from scipy.interpolate import CubicSpline, interp1d

# Fourier Transforms
from jax.numpy.fft import (
    rfft2, irfft2, rfftfreq, fftfreq, ifft2, fftn
)
import pyfftw

# Image Processing
from skimage.transform import downscale_local_mean
from lenspack.image.transforms import starlet2d

# Plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.io import fits
import astropy.units as u

import healpy as hp