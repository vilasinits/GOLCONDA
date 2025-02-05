from utils.config import *


def calculate_histogram_l1norm(image, mask, nbins, density=False):
    """
    Calculates the histogram and L1 norm of an image while accounting for a mask.
    
    Parameters:
        image (np.ndarray): The input image.
        mask (np.ndarray): The mask to apply on the image (1 for valid pixels, 0 to ignore).
        nbins (int or array-like): Number of bins or bin edges for the histogram.
        density (bool): Whether to compute a density-normalized histogram.
    
    Returns:
        binedges (np.ndarray): The edges of the bins.
        bincenters (np.ndarray): The center of each bin.
        hist (np.ndarray): The histogram values for each bin.
        bin_l1_norm (list): The L1 norm for each bin.
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
    
    # Calculate the L1 norm per bin
    bin_l1_norm = [
        np.sum(np.abs(masked_image[digitized == i])) / (total_pixels * bin_width) if density else
        np.sum(np.abs(masked_image[digitized == i]))
        for i in range(1, len(binedges))
    ]
    return np.array(binedges), np.array(bincenters), np.array(hist), np.array(bin_l1_norm)

def adjust_map_l1(input_image, mask, targetvalues, density=False):
    """
    Adjusts wavelet coefficients of an input image to match the target histogram 
    and L1 norm per bin.

    Parameters:
    - input_image (2D array): Input wavelet coefficient map for a given scale.
    - mask (2D array): Boolean mask indicating which coefficients should be modified.
    - targetvalues (dict): Dictionary containing target statistics:
        - 'histogram' (1D array): Target histogram (normalized to sum to 1).
        - 'binedges' (1D array): Bin edges.
        - 'l1_norm' (1D array): Target L1 norm per bin.
        - 'bincenters' (1D array): Bin centers.

    Returns:
    - adjusted_image (2D array): Adjusted wavelet coefficient map.
    """
    # Retrieve target statistics
    bin_edges = targetvalues['binedges']
    
    if density:
        scaling_factor = np.abs(input_image.shape[0]*input_image.shape[1]*(bin_edges[1]-bin_edges[0]))
    else:
        scaling_factor = 1
        
    target_histogram = targetvalues['histogram']*scaling_factor
    target_l1_norm = targetvalues['l1_norm']*scaling_factor
    
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
            index = np.where(coeffs_in_bin < bin_min)[0]
            if len(index) > 0:
                coeffs_in_bin[index] = bin_min
            index = np.where(coeffs_in_bin > bin_max)[0]
            if len(index) >0:
                coeffs_in_bin[index] = bin_max
            # coeffs_in_bin = np.clip(coeffs_in_bin, bin_min, bin_max)
            
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
    Process the target map to compute wavelet coefficients, histograms, L1 norms, and power spectrum.

    Parameters:
        target (np.ndarray): The input map to analyze.
        filter_type (str): The type of filter to use for wavelet decomposition.
        nscales (int): The number of scales for decomposition.
        nbins (list[int]): Number of bins for the histogram at each scale.
        pixelsize (float): Pixel size for power spectrum adjustment.
        adjuster_class: A class implementing compute_power_spectrum().
        decomposer_class: A class implementing decompose().

    Returns:
        dict: A dictionary containing wavelet coefficient statistics and power spectrum information.
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
            'sigma': sigma
        }

    # Add power spectrum information
    # target_values['cls'] = target_cls
    # target_values['ells'] = target_ells
    target_values['coefs'] = target_coefs

    return target_values

def adjust_pixel_values(image, mask, targetvalues, density=False):
    """
    Adjusts the pixel values of an image based on target statistics.

    Args:
        image (ndarray): The input image.
        mask (ndarray): The mask indicating valid pixels.
        targetvalues (dict): The target statistics including 'binedges', 'histogram', and 'l1_norm'.
        density (bool, optional): Flag indicating whether to use density scaling. Defaults to False.

    Returns:
        ndarray: The adjusted image.
        float: The total error normalized by the maximum L1 norm target.

    """
    # Function body...
    # Retrieve target statistics
    binedges_target = targetvalues['binedges']
    if density:
        scaling_factor = np.abs(image.shape[0]*image.shape[1]*(binedges_target[1]-binedges_target[0]))
    else:
        scaling_factor = 1
        
    # scaling_factor = np.abs(image.shape[0]*image.shape[1]*(binedges_target[1]-binedges_target[0]))
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