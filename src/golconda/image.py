# from utils.config import *
import numpy as np
from jax.numpy.fft import ifft2, fftn

def split_image_into_patches(image, num_patches):
    """
    Split a square image into a grid of equal-sized patches.

    Parameters
    ----------
    image : ndarray, shape (N, N)
        Square input image.  ``N`` must be divisible by ``num_patches``.
    num_patches : int
        Number of patches per side, so the total number of patches returned
        is ``num_patches ** 2``.

    Returns
    -------
    list of ndarray
        Patches in row-major order, each of shape
        ``(N // num_patches, N // num_patches)``.
    """
    # Get the height and width of the image
    n = image.shape[0]
    
    # Calculate the patch size (assuming n is divisible by the square root of num_patches)
    patch_size = int( n // num_patches    )
    
    # Check if num_patches is a perfect square and the image can be divided equally
    # if n % patch_size != 0 or num_patches != (n // patch_size) ** 2:
    #     raise ValueError("The image size and number of patches do not match for equal division.")

    # Split the image into patches
    patches = []
    for i in range(0, n, patch_size):
        for j in range(0, n, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    
    return patches

def high_k_taper(k_squared, k_cutoff=0.8):
    """
    Apply a Gaussian taper to suppress high-k noise amplification.

    Parameters
    ----------
    k_squared : ndarray
        Array of squared wavenumber values.
    k_cutoff : float, optional
        Wavenumber cutoff scale.  Frequencies above this are exponentially
        suppressed.  Default is ``0.8``.

    Returns
    -------
    ndarray
        Taper weights in ``[0, 1]``, same shape as ``k_squared``.
    """
    return np.exp(- (k_squared / k_cutoff**2))

def wiener_deconvolution_flat(decomposer, kappa_smooth, radius, reg_param=1e-4, k_cutoff=0.8):
    """
    Perform Wiener deconvolution of a flat-sky convergence map.

    Deconvolves the top-hat smoothing kernel from ``kappa_smooth`` using a
    Wiener filter with Gaussian high-k tapering to prevent noise amplification.

    Parameters
    ----------
    decomposer : WaveletDecomposer
        Decomposer instance used to compute the top-hat Fourier filter and
        the ``k_squared`` grid.
    kappa_smooth : ndarray, shape (N, N)
        Filtered convergence map in real space.
    radius : float
        Radius of the top-hat filter used to smooth the map, in pixel units.
    reg_param : float, optional
        Tikhonov regularisation parameter.  Scaled by ``max(|H|²)`` internally.
        Default is ``1e-4``.
    k_cutoff : float, optional
        Wavenumber cutoff for the Gaussian high-k taper (see
        :func:`high_k_taper`).  Default is ``0.8``.

    Returns
    -------
    ndarray, shape (N, N)
        Deconvolved convergence map.
    """
    # Fourier transform of the smoothed map
    kappa_smooth_ft = fftn(kappa_smooth)

    # Compute k-grid (assuming decomposer has k-squared array)
    decomposer.set_size_dependent_params(kappa_smooth.shape)
    filter_fourier = np.array(decomposer.get_top_hat_filter(radius))  # Convert JAX to NumPy

    # Compute Wiener filter with adaptive regularization
    epsilon = reg_param * np.max(filter_fourier**2)
    wiener_filter = filter_fourier / (filter_fourier**2 + epsilon)

    # Apply high-k tapering
    k_squared = decomposer.k_squared
    taper = high_k_taper(k_squared, k_cutoff)
    wiener_filter *= taper  # Smooth high-frequency components

    # Apply Wiener filter in Fourier space
    kappa_deconv_ft = kappa_smooth_ft * wiener_filter

    # Transform back to real space
    kappa_deconv = np.real(ifft2(kappa_deconv_ft))

    return kappa_deconv

def compute_binedges(bincenters):
    """
    Compute bin edges from an array of bin centres.

    Parameters
    ----------
    bincenters : ndarray, shape (B,)
        Uniformly or non-uniformly spaced bin centre values.

    Returns
    -------
    edges : ndarray, shape (B+1,)
        Bin edges such that ``edges[i]`` and ``edges[i+1]`` bound
        ``bincenters[i]``.
    """
    # Calculate bin widths (differences between consecutive bin centers)
    bin_widths = np.diff(bincenters)

    # Initialize an array for edges
    edges = np.zeros(len(bincenters) + 1)
    edges[1:-1] = bincenters[:-1] + bin_widths / 2  # Midpoints between centers

    # Add the first and last edges
    edges[0] = bincenters[0] - 0.5 * bin_widths[0]
    edges[-1] = bincenters[-1] + 0.5 * bin_widths[-1]

    return edges