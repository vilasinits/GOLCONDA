# from utils.config import *
import numpy as np
import jax
import jax.numpy as jnp


def calculate_histogram_l1norm(image, mask, nbins, density=False):
    """
    Calculate the histogram and per-bin ℓ₁ norm of an image with optional masking.

    The ℓ₁ norm is always stored normalised as
    ``Σ|x in bin| / (total_pixels × bin_width)`` — equivalent to
    ``PDF(bin_centre) × |bin_centre|`` — so that it is portable across maps
    of different sizes.

    Parameters
    ----------
    image : ndarray, shape (N, M)
        Input image.
    mask : ndarray or None
        Boolean mask; pixels where ``mask == 1`` are included.  Pass ``None``
        to include all pixels.
    nbins : int or array-like
        Number of equally-spaced bins, or an explicit array of bin edges.
    density : bool, optional
        If ``True`` the histogram is returned as a probability density
        (``counts / (total × bin_width)``).  Default is ``False`` (raw counts).

    Returns
    -------
    binedges : ndarray, shape (nbins+1,)
        Edges of the histogram bins.
    bincenters : ndarray, shape (nbins,)
        Centre of each bin.
    hist : ndarray, shape (nbins,)
        Histogram values (raw counts or density depending on ``density``).
    bin_l1_norm : ndarray, shape (nbins,)
        Per-bin ℓ₁ norm, normalised per pixel per bin width.
    """
    # If no mask is provided, assume all pixels are valid
    if mask is None:
        mask = np.ones_like(image, dtype=bool)
        
    # Apply the mask to filter out invalid pixels
    masked_image = image[mask==1]

    # Define bin edges and centers
    if np.ndim(nbins) > 0:  # If nbins is an array, treat it as bin edges
        binedges = np.asarray(nbins)
    else:  # Otherwise, generate bin edges using the range of the masked image
        binedges = np.linspace(np.min(masked_image), np.max(masked_image), nbins + 1)
    
    bincenters = 0.5 * (binedges[:-1] + binedges[1:])
    
    # Calculate the histogram using only the masked pixels
    hist, _ = np.histogram(masked_image, bins=binedges, density=density)
    
    # Calculate the bin width (for normalization if density=True)
    bin_width = binedges[1] - binedges[0]
    total_pixels = np.prod(image.shape)

    # Digitize the masked image to determine bin membership
    digitized = np.digitize(masked_image, binedges, right=False)

    # L1 norm is always normalised per pixel per bin-width so it is portable
    # across maps of different sizes.  When density=False the histogram is raw
    # counts, but we still normalise l1_norm the same way so that adjust_map_l1
    # can rescale it correctly for any output map size.
    bin_l1_norm = [
        np.sum(np.abs(masked_image[digitized == i])) / (total_pixels * bin_width)
        for i in range(1, len(binedges))
    ]
    return np.array(binedges), np.array(bincenters), np.array(hist), np.array(bin_l1_norm)

def adjust_map_l1(input_image, mask, targetvalues, density=False):
    """
    Adjust wavelet coefficients to match a target per-bin histogram and ℓ₁ norm.

    Pixels are sorted, assigned to bins according to ``targetvalues['histogram']``,
    and shifted within each bin so that the per-bin ℓ₁ norm matches
    ``targetvalues['l1_norm']``.  Positive and negative coefficients are adjusted
    in two passes to preserve sign structure, then clipped to the bin boundaries.

    Parameters
    ----------
    input_image : ndarray, shape (N, M)
        Input wavelet coefficient map for a single decomposition scale.
    mask : ndarray or None
        Boolean mask; only pixels where ``mask != 0`` are modified.  Pass
        ``None`` to process the entire map.
    targetvalues : dict
        Target statistics, as produced by :func:`process_image`.  Required
        keys: ``'binedges'``, ``'histogram'``, ``'l1_norm'``.  Optional key:
        ``'source_pixels'`` (total pixels in the source map; used to rescale
        raw histogram counts when ``density=False``).
    density : bool, optional
        Set ``True`` when ``targetvalues['histogram']`` is a probability
        density rather than raw counts.  Default is ``False``.

    Returns
    -------
    adjusted_image : ndarray, shape (N, M)
        Copy of ``input_image`` with masked pixels adjusted.
    normalised_l1_error : float
        Total absolute ℓ₁ error across all bins, normalised by the maximum
        per-bin target ℓ₁ norm.
    """
    # Retrieve target statistics
    bin_edges = targetvalues['binedges']
    bin_width = bin_edges[1] - bin_edges[0]
    output_pixels = input_image.shape[0] * input_image.shape[1]

    # l1_norm is always stored normalised per-pixel per-bin-width (see
    # calculate_histogram_l1norm), so rescale to absolute counts for this
    # output map regardless of density flag.
    l1_scaling = output_pixels * bin_width
    target_l1_norm = targetvalues['l1_norm'] * l1_scaling

    if density:
        # histogram is a probability density (counts / total / bin_width), rescale to counts
        target_histogram = targetvalues['histogram'] * l1_scaling
    else:
        # histogram is raw counts from the source map; rescale to this output map's pixel count.
        source_pixels = targetvalues.get('source_pixels', output_pixels)
        target_histogram = targetvalues['histogram'] * (output_pixels / source_pixels)
    
    if mask is None:
        mask = np.ones_like(input_image)
        
    # Extract coefficients using the mask
    totpixels = np.prod(input_image.shape)
    totmaskedpixels = np.count_nonzero(mask)
    mask_indices = np.nonzero(mask)  # Indices of valid (non-zero mask) pixels

    ratio_masked_field = totmaskedpixels / totpixels
    
    input_masked = input_image[mask_indices]
    sorted_indices = np.argsort(input_masked)
    
    start_idx = 0
    num_bins = len(target_histogram)
    
    total_l1_error = 0.0
    for bin_idx in range(num_bins):
        # Compute the number of coefficients to allocate to this bin
        num_coeffs_in_bin = round(target_histogram[bin_idx] * ratio_masked_field)
        end_idx = start_idx + num_coeffs_in_bin

        # Ensure the last bin includes all remaining coefficients
        if bin_idx == num_bins - 1:
            end_idx = totmaskedpixels
        elif end_idx > totmaskedpixels:
            end_idx = totmaskedpixels

        # Get bin boundaries and target L1 norm
        bin_min = bin_edges[bin_idx]
        bin_max = bin_edges[bin_idx + 1]
        target_l1_bin = target_l1_norm[bin_idx] * ratio_masked_field

        if num_coeffs_in_bin > 0 and end_idx > start_idx:
            start_idx = int(start_idx)
            end_idx = int(end_idx)

            # Extract coefficients in this bin
            coeffs_in_bin = input_masked[sorted_indices[start_idx:end_idx]]

            # Compute current L1 norm in this bin
            current_l1_bin = np.sum(np.abs(coeffs_in_bin))
            delta_bin_value = (target_l1_bin - current_l1_bin) / num_coeffs_in_bin

            # Adjust positive values
            pos_indices = np.where(coeffs_in_bin >= 0)[0]
            if len(pos_indices) > 0:
                coeffs_in_bin[pos_indices] += delta_bin_value
                coeffs_in_bin[pos_indices] = np.maximum(coeffs_in_bin[pos_indices], 0)

            # Recalculate L1 norm
            current_l1_bin = np.sum(np.abs(coeffs_in_bin))

            # Adjust negative values
            neg_indices = np.where(coeffs_in_bin < 0)[0]
            if len(neg_indices) > 0:
                neg_adjustment = (target_l1_bin - current_l1_bin) / len(neg_indices)
                coeffs_in_bin[neg_indices] -= neg_adjustment
                coeffs_in_bin[neg_indices] = np.minimum(coeffs_in_bin[neg_indices], 0)

            # Enforce final boundary constraints
            # index = np.where(coeffs_in_bin < bin_min)[0]
            # if len(index) > 0:
            #     coeffs_in_bin[index] = bin_min
            # index = np.where(coeffs_in_bin > bin_max)[0]
            # if len(index) >0:
            #     coeffs_in_bin[index] = bin_max
            coeffs_in_bin = np.clip(coeffs_in_bin, bin_min, bin_max)
            
            # Compute L1 norm error for diagnostics
            current_l1_bin = np.sum(np.abs(coeffs_in_bin))
            total_l1_error += np.abs(target_l1_bin - current_l1_bin)

            # Update sorted coefficients
            input_masked[sorted_indices[start_idx:end_idx]] = coeffs_in_bin

        start_idx += num_coeffs_in_bin

    # Restore the adjusted coefficients into the original image
    adjusted_image = np.copy(input_image)
    adjusted_image[mask_indices] = input_masked  

    return adjusted_image, total_l1_error /np.max(target_l1_norm)


def process_image(target, filter_type, nscales, nbins, decomposer_class, density=False):
    """
    Extract per-scale wavelet statistics from a target map.

    Decomposes ``target`` into ``nscales + 1`` wavelet bands and computes the
    histogram and ℓ₁ norm for each band.  The returned dictionary is used
    directly as the target for :func:`adjust_map_l1` during emulation.

    Parameters
    ----------
    target : ndarray, shape (N, M)
        Input convergence map to analyse.
    filter_type : {'tophat', 'starlet'}
        Wavelet filter type passed to :meth:`~golconda.WaveletDecomposer.decompose`.
    nscales : int
        Number of wavelet scales.  The coarse residual is stored as
        ``'scale_{nscales}'``.
    nbins : list of int, length nscales+1
        Number of histogram bins at each scale (index ``j`` → scale ``j``).
    decomposer_class : WaveletDecomposer
        Initialised decomposer instance.
    density : bool, optional
        Passed to :func:`calculate_histogram_l1norm`.  Default is ``False``.

    Returns
    -------
    target_values : dict
        Dictionary with keys ``'scale_0'``, …, ``'scale_{nscales}'`` and
        ``'coefs'``.  Each scale entry contains ``'histogram'``, ``'binedges'``,
        ``'l1_norm'``, ``'bincenters'``, ``'sigma'``, and ``'source_pixels'``.
    """
    target_values = {}

    # Compute the power spectrum of the target map
    # target_ells, target_cls = adjuster_class.compute_power_spectrum(target)

    # Decompose the target map into wavelet coefficients
    target_coefs = decomposer_class.decompose(
        target, num_scales=nscales, filter_type=filter_type, recalculate_params=True
    )

    # Process each scale and compute histograms, L1 norms, etc.
    for scale in range(nscales + 1):
        sigma = np.std(target_coefs[scale])
        edges, centers, hist, l1_norm = calculate_histogram_l1norm(
            target_coefs[scale], None, nbins[scale], density=density
        )
        target_values[f'scale_{scale}'] = {
            'histogram': hist,
            'binedges': edges,
            'l1_norm': l1_norm,
            'bincenters': centers,
            'sigma': sigma,
            'source_pixels': target_coefs[scale].size,
        }

    # Add power spectrum information
    # target_values['cls'] = target_cls
    # target_values['ells'] = target_ells
    target_values['coefs'] = target_coefs

    return target_values

def adjust_pixel_values(image, mask, targetvalues, density=False):
    """
    Adjust pixel values of an image to match target histogram and ℓ₁-norm statistics.

    Similar to :func:`adjust_map_l1` but uses proportional scaling (rather than
    additive shifts) to match the target ℓ₁ norm within each bin, keeping
    positive and negative contributions separately scaled.

    Parameters
    ----------
    image : ndarray, shape (N, M)
        Input image whose pixel values will be adjusted in-place.
    mask : ndarray or None
        Boolean mask; only pixels where ``mask != 0`` are modified.
    targetvalues : dict
        Target statistics with keys ``'binedges'``, ``'histogram'``, and
        ``'l1_norm'``.
    density : bool, optional
        If ``True``, ``histogram`` and ``l1_norm`` are treated as probability
        densities and rescaled to absolute counts.  Default is ``False``.

    Returns
    -------
    image : ndarray, shape (N, M)
        Adjusted image (modified in-place and returned).
    normalised_error : float
        Total absolute ℓ₁ error normalised by the maximum per-bin target.
    """
    # Retrieve target statistics
    binedges_target = targetvalues['binedges']
    if density:
        scaling_factor = np.abs(image.shape[0]*image.shape[1]*(binedges_target[1]-binedges_target[0]))
    else:
        scaling_factor = 1
        
    histogram_target = targetvalues['histogram']*scaling_factor
    l1_norm_target = targetvalues['l1_norm']*scaling_factor
    
    nbins = len(histogram_target)
    if mask is None:
        mask = np.ones_like(image)
    # Extract masked pixel values
    mask_indices = np.nonzero(mask)  # Indices of valid (non-zero mask) pixels
    masked_pixel_values = image[mask_indices]
    sorted_pixel_indices = np.argsort(masked_pixel_values)  # Indices of sorted masked values

    # Initialize binning
    total_pixels = image.size  # Total number of pixels in the image
    total_masked_pixels = len(masked_pixel_values)  # Total number of valid (masked) pixels
    pixel_fraction = total_masked_pixels / total_pixels  # Fraction of valid pixels
    start_index = 0  # Start index for the current bin
    total_error = 0  # Track total error

    # Iterate over bins
    for bin_index in range(nbins):
        # Compute expected number of pixels in the current bin
        expected_pixels_in_bin = histogram_target[bin_index] * pixel_fraction
        end_index = start_index + expected_pixels_in_bin

        # Handle edge cases for the last bin and invalid end_index
        if bin_index == nbins - 1 or end_index > total_masked_pixels:
            end_index = total_masked_pixels

        target_l1_norm = l1_norm_target[bin_index] * pixel_fraction

        # Process pixels in the bin if there are any
        if expected_pixels_in_bin > 0 and end_index > start_index:
            start_index = int(start_index)
            end_index = int(end_index)

            # Extract pixel values in this bin
            bin_pixel_values = masked_pixel_values[sorted_pixel_indices[start_index:end_index]]

            # Apply L1 norm adjustment
            current_l1_norm = np.sum(np.abs(bin_pixel_values))
            if current_l1_norm > 0:
                # Separate positive and negative coefficients
                positive_indices = np.where(bin_pixel_values > 0)[0]
                negative_indices = np.where(bin_pixel_values < 0)[0]

                # Proportional scaling for positive and negative coefficients
                positive_sum = np.sum(bin_pixel_values[positive_indices])
                negative_sum = np.sum(np.abs(bin_pixel_values[negative_indices]))
                total_sum = positive_sum + negative_sum

                if positive_sum > 0:
                    scaling_factor_positive = target_l1_norm * (positive_sum / total_sum) / positive_sum
                    bin_pixel_values[positive_indices] *= scaling_factor_positive

                positive_indices = np.where(bin_pixel_values > 0)[0]
                negative_indices = np.where(bin_pixel_values < 0)[0]
                positive_sum = np.sum(bin_pixel_values[positive_indices])
                negative_sum = np.sum(np.abs(bin_pixel_values[negative_indices]))
                
                if negative_sum > 0:
                    scaling_factor_negative = target_l1_norm * (negative_sum / total_sum) / negative_sum
                    bin_pixel_values[negative_indices] *= scaling_factor_negative

            # Apply boundary constraints
            bin_pixel_values = np.clip(bin_pixel_values, binedges_target[bin_index], binedges_target[bin_index + 1])

            # Calculate error for this bin
            adjusted_l1_norm = np.sum(np.abs(bin_pixel_values))
            bin_error = np.abs(target_l1_norm - adjusted_l1_norm) 
            total_error += bin_error

            # Update the masked pixel values
            masked_pixel_values[sorted_pixel_indices[start_index:end_index]] = bin_pixel_values

        # Update the start index for the next bin
        start_index += expected_pixels_in_bin

    # Write back the adjusted pixel values to the original image
    image[mask_indices] = masked_pixel_values
    return image, total_error/np.max(l1_norm_target)


# ---------------------------------------------------------------------------
# JAX-accelerated L1 adjustment (GPU-ready)
# ---------------------------------------------------------------------------

@jax.jit
def _adjust_l1_kernel(
    sorted_vals: jnp.ndarray,
    pixel_bin: jnp.ndarray,
    target_l1_per_bin: jnp.ndarray,
    bin_edges: jnp.ndarray,
) -> jnp.ndarray:
    """
    Vectorised per-pixel L1 adjustment, JIT-compiled for GPU execution.

    Replaces the Python bin loop in :func:`adjust_map_l1` with scatter-add
    aggregates and element-wise operations that run entirely on the JAX device.

    Parameters
    ----------
    sorted_vals : (M,) float array
        Masked pixel values in ascending order.
    pixel_bin : (M,) int array
        Bin index for each sorted pixel.
    target_l1_per_bin : (B,) float array
        Target L1 norm for each bin (scaled to output map).
    bin_edges : (B+1,) float array
        Bin edge values for clipping.

    Returns
    -------
    jnp.ndarray, shape (M,)
        Adjusted sorted pixel values.
    """
    num_bins = target_l1_per_bin.shape[0]
    pos_mask = sorted_vals >= 0.0

    # Actual pixel count per bin (may differ from target histogram due to rounding)
    actual_counts = jnp.zeros(num_bins).at[pixel_bin].add(1.0)

    # Pass 1 — adjust positive values upward
    current_l1 = jnp.zeros(num_bins).at[pixel_bin].add(jnp.abs(sorted_vals))
    delta_per_bin = (target_l1_per_bin - current_l1) / jnp.maximum(actual_counts, 1.0)
    vals = jnp.where(
        pos_mask,
        jnp.maximum(sorted_vals + delta_per_bin[pixel_bin], 0.0),
        sorted_vals,
    )

    # Pass 2 — adjust negative values with remaining L1 budget
    current_l1_2 = jnp.zeros(num_bins).at[pixel_bin].add(jnp.abs(vals))
    remaining_l1 = target_l1_per_bin - current_l1_2
    neg_counts = jnp.zeros(num_bins).at[pixel_bin].add((~pos_mask).astype(jnp.float32))
    neg_adj_per_bin = remaining_l1 / jnp.maximum(neg_counts, 1.0)
    vals = jnp.where(
        ~pos_mask,
        jnp.minimum(vals - neg_adj_per_bin[pixel_bin], 0.0),
        vals,
    )

    # Clip to bin range
    return jnp.clip(vals, bin_edges[pixel_bin], bin_edges[pixel_bin + 1])


def adjust_map_l1_jax(input_image, mask, targetvalues, density=False):
    """
    JAX-accelerated version of :func:`adjust_map_l1`.

    Replaces the per-bin Python loop with a fully vectorised kernel compiled
    with ``@jax.jit``.  When JAX is configured for GPU (e.g.
    ``pip install "jax[cuda12]"``), the sort, scatter-add aggregates, and all
    per-pixel adjustments run on the device.

    Parameters
    ----------
    input_image : ndarray, shape (N, M)
        Input wavelet coefficient map for a given scale.
    mask : ndarray or None
        Boolean mask (1 = valid pixel).
    targetvalues : dict
        Same format as :func:`adjust_map_l1`:
        ``'binedges'``, ``'histogram'``, ``'l1_norm'``, ``'source_pixels'``.
    density : bool
        Whether the stored histogram is a probability density.

    Returns
    -------
    adjusted_image : ndarray
    normalised_l1_error : float
    """
    bin_edges = targetvalues['binedges']
    bin_width = bin_edges[1] - bin_edges[0]
    output_pixels = input_image.shape[0] * input_image.shape[1]
    num_bins = len(targetvalues['l1_norm'])

    l1_scaling = output_pixels * bin_width
    target_l1_norm = targetvalues['l1_norm'] * l1_scaling

    if density:
        target_histogram = targetvalues['histogram'] * l1_scaling
    else:
        source_pixels = targetvalues.get('source_pixels', output_pixels)
        target_histogram = targetvalues['histogram'] * (output_pixels / source_pixels)

    if mask is None:
        mask = np.ones_like(input_image)

    totpixels = np.prod(input_image.shape)
    totmaskedpixels = int(np.count_nonzero(mask))
    mask_indices = np.nonzero(mask)
    ratio_masked_field = totmaskedpixels / totpixels

    # Sort on JAX device (GPU when available)
    input_masked = jnp.asarray(input_image[mask_indices])
    sorted_indices = jnp.argsort(input_masked)
    sorted_vals = input_masked[sorted_indices]

    # Assign each rank to a bin using cumulative bin-size boundaries
    bin_counts = jnp.round(
        jnp.asarray(target_histogram * ratio_masked_field)
    ).astype(jnp.int32)
    cumulative = jnp.concatenate(
        [jnp.array([0], dtype=jnp.int32), jnp.cumsum(bin_counts)]
    )
    # searchsorted on the interior breakpoints gives the bin index for each rank
    ranks = jnp.arange(totmaskedpixels, dtype=jnp.int32)
    pixel_bin = jnp.minimum(
        jnp.searchsorted(cumulative[1:-1], ranks, side='right'),
        num_bins - 1,
    )

    target_l1_per_bin = jnp.asarray(target_l1_norm * ratio_masked_field)
    bin_edges_jax = jnp.asarray(bin_edges)

    # Run JIT-compiled kernel
    adjusted_sorted = _adjust_l1_kernel(
        sorted_vals, pixel_bin, target_l1_per_bin, bin_edges_jax
    )

    # Unsort: scatter adjusted values back to their original positions
    adjusted_masked = jnp.zeros_like(input_masked).at[sorted_indices].set(adjusted_sorted)

    # Diagnostic error
    adjusted_l1_per_bin = jnp.zeros(num_bins).at[pixel_bin].add(jnp.abs(adjusted_sorted))
    total_l1_error = float(jnp.sum(jnp.abs(target_l1_per_bin - adjusted_l1_per_bin)))
    normalised_error = total_l1_error / float(jnp.maximum(jnp.max(target_l1_per_bin), 1e-12))

    # Write back to a writable CPU array
    adjusted_image = np.copy(input_image)
    adjusted_image[mask_indices] = np.array(adjusted_masked)

    return adjusted_image, normalised_error