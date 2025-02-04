import numpy as np
from lenspack.image.transforms import starlet2d
from jax.numpy.fft import ifft2, fftn, fftfreq
import scipy
from typing import List

import jax.numpy as jnp

class WaveletDecomposer:
    def __init__(self, num_scales: int = 3) -> None:
        """
        Initialize the WaveletDecomposer class with optional static parameter initialization.

        Parameters:
        -----------
        num_scales : int, optional
            The number of decomposition scales to be used. Defaults to 3.
        """
        self.num_scales = num_scales
        self.precomputed_tophat_filters = {}

    def set_size_dependent_params(self, image_shape, L = None):
        """
        Initialize size-dependent parameters (frequency grids, pixel size, etc.) based on the image shape.

        Parameters:
        -----------
        image_shape : Tuple[int, int]
            The shape of the input image.
        L : int, optional
            The length scale of the map. Defaults to the size of the input map.
        """
        self.image_shape = image_shape
        self.L = L or image_shape[0]  # Set the length scale to the size of the image if not provided
        self.dx = self.L / image_shape[0]  # Calculate pixel size
        self.kx, self.ky = fftfreq(image_shape[0], self.dx), fftfreq(image_shape[1], self.dx)
        self.kx, self.ky = jnp.meshgrid(self.kx, self.ky, indexing='ij')
        self.k_squared = (self.kx**2 + self.ky**2)
        self.precomputed_tophat_filters.clear()

    def set_image(self, input_map: np.ndarray) -> None:
        """
        Set the input image and compute its Fourier transform.

        Parameters:
        -----------
        input_map : np.ndarray
            The new input image or map for decomposition.
        """
        self.input_map = input_map
        self.field_ft = fftn(self.input_map)  # Compute the Fourier transform of the input map

    @staticmethod
    def top_hat_window_fourier(k: np.ndarray) -> np.ndarray:
        """
        Compute the top-hat window function in Fourier space.

        Parameters:
        -----------
        k : np.ndarray
            The frequency grid in Fourier space.

        Returns:
        --------
        np.ndarray
            The top-hat filter in Fourier space.
        """
        k = jnp.where(k == 0, 1e-7, k)  # Avoid division by zero
        return 2.0 * scipy.special.j1(k) / k

    def get_top_hat_filter(self, window_radius: float) -> np.ndarray:
        """
        Retrieve or compute the top-hat filter in Fourier space for a given radius.

        Parameters:
        -----------
        window_radius : float
            The radius for the top-hat filter in Fourier space.

        Returns:
        --------
        np.ndarray
            The top-hat filter for the specified radius.
        """
        k = 2 * jnp.pi * jnp.sqrt(self.k_squared)
        filter_window = self.top_hat_window_fourier(k * window_radius)
        return filter_window

    def decompose_with_tophat(self) -> np.ndarray:
        """
        Perform wavelet decomposition using the top-hat filter.

        Returns:
        --------
        np.ndarray
            The wavelet coefficients and coarse scale image.
        """
        coarse_image = self.input_map  # Coarse scale image (initially the input map)
        wavelet_coeffs = []  # List to store wavelet coefficients
        for scale in range(1, self.num_scales + 1):
            # Apply the top-hat filter in Fourier space for the current scale (2^scale)
            coarse_image_ft = self.field_ft * self.get_top_hat_filter(2**scale)
            coarse_image_new = ifft2(coarse_image_ft).real  # Inverse Fourier transform to obtain the new coarse image
            wavelet_coeff = coarse_image - coarse_image_new  # Compute the wavelet coefficient
            coarse_image = coarse_image_new  # Update coarse image for the next iteration
            wavelet_coeffs.append(wavelet_coeff)
        wavelet_coeffs.append(coarse_image)  # Append the final coarse image
        return np.array(wavelet_coeffs)

    def decompose_with_starlet(self) -> np.ndarray:
        """
        Perform wavelet decomposition using the starlet transform.

        Returns:
        --------
        np.ndarray
            The wavelet coefficients and coarse scale image.
        """
        return starlet2d(self.input_map, self.num_scales)

    def reconstruct(self, coefficients: List[np.ndarray]) -> np.ndarray:
        """
        Reconstruct image from wavelet coefficients by summing all scales.

        Args:
            coefficients: List of coefficient arrays from decompose()

        Returns:
            Reconstructed image array
        """
        # Simple reconstruction by summing all scales
        return np.sum(coefficients, axis=0)

    def decompose(self, input_map: np.ndarray, num_scales = 1, filter_type: str = 'tophat', recalculate_params: bool = True):
        """
        Perform wavelet decomposition based on the chosen filter type for the given image.

        Parameters:
        -----------
        input_map : np.ndarray
            The image to decompose.
        num_scales : int, optional
            Number of scales for decomposition. If None, the class default is used.
        filter_type : str, optional
            The type of filter to use for decomposition ('tophat' or 'starlet'). Default is 'tophat'.
        recalculate_params : bool, optional
            If True, recalculates the static parameters (L, kx, ky) based on the image size. Default is False.

        Returns:
        --------
        np.ndarray
            The wavelet coefficients and coarse scale image.
        """
        if num_scales is not None:
            self.num_scales = num_scales

        if recalculate_params:
            self.set_size_dependent_params(input_map.shape)

        self.set_image(input_map)

        if filter_type == 'tophat':
            return self.decompose_with_tophat()
        elif filter_type == 'starlet':
            return self.decompose_with_starlet()
        else:
            raise ValueError(f"Unknown filter type: {filter_type}. Supported values are 'tophat' and 'starlet'.")


if __name__ == "__main__":
    input_image_1 = np.random.rand(256, 256)
    input_image_2 = np.random.rand(256, 256)

    decomposer = WaveletDecomposer()

    wavelet_coefficients_1 = decomposer.decompose(input_image_1, num_scales=5, filter_type='tophat', recalculate_params=True)

    wavelet_coefficients_2 = decomposer.decompose(input_image_2, num_scales=5, filter_type='starlet', recalculate_params=False)


# import numpy as np
# from lenspack.image.transforms import starlet2d
# from jax.numpy.fft import ifft2, fftn, fftfreq
# import jax.numpy as jnp
# import scipy
# from typing import List

# class WaveletDecomposer:
#     """
#     Class to perform wavelet decomposition using different filters (e.g., top-hat, starlet),
#     allowing the user to control whether static parameters should be recalculated.
    
#     Attributes:
#     -----------
#     num_scales : int
#         The number of decomposition scales to be used.
#     L : int
#         The characteristic length scale, typically set to the size of the input map.
#     dx : float
#         The pixel size derived from the input map and length scale.
#     kx, ky : ndarray
#         The frequency grids used for the Fourier transform in both x and y directions.
#     precomputed_tophat_filters : dict
#         Cache to store precomputed top-hat filters for different scales.
#     """

#     def __init__(self, num_scales=3):
#         """
#         Initialize the WaveletDecomposer class with optional static parameter initialization.
        
#         Parameters:
#         -----------
#         num_scales : int, optional
#             The number of decomposition scales to be used. Defaults to 3.
#         """
#         self.num_scales = num_scales
#         self.precomputed_tophat_filters = {}
#         # self.initialized = False  # Keeps track if the static parameters are initialized

#     def set_size_dependent_params(self, image_shape, L=None):
#         """
#         Initialize size-dependent parameters (frequency grids, pixel size, etc.) based on the image shape.
        
#         Parameters:
#         -----------
#         image_shape : tuple
#             The shape of the input image.
#         L : int, optional
#             The length scale of the map. Defaults to the size of the input map.
#         """
#         self.image_shape = image_shape
#         self.L =  L or image_shape[0]  # Set the length scale to the size of the image if not provided
#         self.dx = self.L / image_shape[0]  # Calculate pixel size
#         self.kx, self.ky = fftfreq(image_shape[0], self.dx), fftfreq(image_shape[1], self.dx)
#         self.kx, self.ky = jnp.meshgrid(self.kx, self.ky, indexing='ij')
#         self.k_squared = (self.kx**2 + self.ky**2)
#         self.precomputed_tophat_filters.clear()  # Clear cache when image size changes
#         # self.initialized = True  # Mark static parameters as initialized

#     def set_image(self, input_map):
#         """
#         Set the input image and compute its Fourier transform.
        
#         Parameters:
#         -----------
#         input_map : ndarray
#             The new input image or map for decomposition.
#         """
#         self.input_map = input_map
#         self.field_ft = fftn(self.input_map)  # Compute the Fourier transform of the input map

#     @staticmethod
#     def top_hat_window_fourier(k):
#         """
#         Compute the top-hat window function in Fourier space.
        
#         Parameters:
#         -----------
#         k : ndarray
#             The frequency grid in Fourier space.
        
#         Returns:
#         --------
#         ndarray
#             The top-hat filter in Fourier space.
#         """
#         k = jnp.where(k == 0, 1e-7, k)  # Avoid division by zero
#         return 2.0 * scipy.special.j1(k) / k

#     def get_top_hat_filter(self, window_radius):
#         """
#         Retrieve or compute the top-hat filter in Fourier space for a given radius.
        
#         Parameters:
#         -----------
#         window_radius : float
#             The radius for the top-hat filter in Fourier space.
        
#         Returns:
#         --------
#         ndarray
#             The top-hat filter for the specified radius.
#         """
#         # if window_radius in self.precomputed_tophat_filters:
#         #     return self.precomputed_tophat_filters[window_radius]
        
#         k = 2 * jnp.pi * jnp.sqrt(self.k_squared)
#         filter_window = self.top_hat_window_fourier(k * window_radius)
#         # self.precomputed_tophat_filters[window_radius] = filter_window  # Cache the filter for future use
#         return filter_window

#     def decompose_with_tophat(self):
#         """
#         Perform wavelet decomposition using the top-hat filter.
        
#         Returns:
#         --------
#         ndarray
#             The wavelet coefficients and coarse scale image.
#         """
#         coarse_image = self.input_map  # Coarse scale image (initially the input map)
#         wavelet_coeffs = []  # List to store wavelet coefficients

#         for scale in range(1, self.num_scales + 1):
#             # Apply the top-hat filter in Fourier space for the current scale (2^scale)
#             coarse_image_ft = self.field_ft * self.get_top_hat_filter(2**scale)
#             coarse_image_new = ifft2(coarse_image_ft).real  # Inverse Fourier transform to obtain the new coarse image
#             wavelet_coeff = coarse_image - coarse_image_new  # Compute the wavelet coefficient
#             coarse_image = coarse_image_new  # Update coarse image for the next iteration
#             wavelet_coeffs.append(wavelet_coeff)

#         wavelet_coeffs.append(coarse_image)  # Append the final coarse image
#         return np.array(wavelet_coeffs)

#     def decompose_with_starlet(self):
#         """
#         Perform wavelet decomposition using the starlet transform.
        
#         Returns:
#         --------
#         ndarray
#             The wavelet coefficients and coarse scale image.
#         """
#         return starlet2d(self.input_map, self.num_scales)
    
#     def reconstruct(self, coefficients):
#         """
#         Reconstruct image from wavelet coefficients by summing all scales.
        
#         Args:
#             coefficients: List of coefficient arrays from decompose()
            
#         Returns:
#             Reconstructed image array
#         """
#         # Simple reconstruction by summing all scales
#         return np.sum(coefficients, axis=0)

#     def decompose(self, input_map, num_scales=None, filter_type='tophat', recalculate_params=True):
#         """
#         Perform wavelet decomposition based on the chosen filter type for the given image.
        
#         Parameters:
#         -----------
#         input_map : ndarray
#             The image to decompose.
#         num_scales : int, optional
#             Number of scales for decomposition. If None, the class default is used.
#         filter_type : str, optional
#             The type of filter to use for decomposition ('tophat' or 'starlet'). Default is 'tophat'.
#         recalculate_params : bool, optional
#             If True, recalculates the static parameters (L, kx, ky) based on the image size. Default is False.
        
#         Returns:
#         --------
#         ndarray
#             The wavelet coefficients and coarse scale image.
#         """
#         # Set number of scales if provided
#         if num_scales is not None:
#             self.num_scales = num_scales

#         # Check if we need to recalculate the static parameters
#         if recalculate_params: #or not self.initialized:
#             self.set_size_dependent_params(input_map.shape)

#         # Set the new image and compute its Fourier transform
#         self.set_image(input_map)

#         # Perform the decomposition based on the filter type
#         if filter_type == 'tophat':
#             return self.decompose_with_tophat()
#         elif filter_type == 'starlet':
#             return self.decompose_with_starlet()
#         else:
#             raise ValueError(f"Unknown filter type: {filter_type}. Supported values are 'tophat' and 'starlet'.")


# # Example usage:
# if __name__ == "__main__":
#     # Example input images
#     input_image_1 = np.random.rand(256, 256)  # First input image
#     input_image_2 = np.random.rand(256, 256)  # Second input image (same size)

#     # Initialize the decomposer
#     decomposer = WaveletDecomposer()

#     # First image, recalculate parameters (since it's the first run)
#     wavelet_coefficients_1 = decomposer.decompose(input_image_1, num_scales=5, filter_type='tophat', recalculate_params=True)

#     # Second image, do not recalculate parameters (same size as the first)
#     wavelet_coefficients_2 = decomposer.decompose(input_image_2, num_scales=5, filter_type='starlet', recalculate_params=False)
