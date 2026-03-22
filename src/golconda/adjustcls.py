# from utils.config import *
import numpy as np
import jax
import jax.numpy as jnp
import pyfftw
from jax.numpy.fft import rfft2, irfft2, rfftfreq, fftfreq, ifft2, fftn
from scipy.interpolate import interp1d
class PowerSpectrumAdjuster:
    """
    Adjust a flat-sky map so its angular power spectrum matches a target.

    Uses pyFFTW for forward/inverse FFTs and scales Fourier amplitudes by
    ``sqrt(C_ell_target / C_ell_input)`` while preserving phases.

    Parameters
    ----------
    map_shape : tuple of int
        Shape ``(N, N)`` of the input map (square maps assumed).
    pixel_size : float
        Pixel size in arcminutes per pixel.

    Attributes
    ----------
    pix_to_rad : float
        Pixel size converted to radians per pixel.
    ell2d : ndarray
        2-D multipole-moment grid.
    radial_bins : ndarray
        Precomputed integer radial-bin indices used for binning.
    """

    def __init__(self, map_shape, pixel_size):
        """
        Parameters
        ----------
        map_shape : tuple of int
            Shape ``(N, N)`` of the input map.
        pixel_size : float
            Pixel size in arcminutes per pixel.
        """
        self.map_shape = map_shape
        self.pixel_size = pixel_size
        self.pix_to_rad = pixel_size / 60. * np.pi / 180.  # Convert to radians/pixel
        self.N = map_shape[0]  # Assume square maps
        self.center = self.N / 2

        # Enable pyFFTW plan cache for efficiency
        pyfftw.interfaces.cache.enable()

        # Precompute 2D ell grid and radial bins
        self._precompute_grids()

    def _precompute_grids(self):
        """Precompute the 2-D ℓ grid and integer radial-bin indices."""
        inds = jnp.fft.fftfreq(self.N, d=1. / self.N)
        X, Y = jnp.meshgrid(inds, inds)
        self.R = jnp.sqrt(X**2 + Y**2)
        self.ell2d = self.R * (2 * np.pi / self.pix_to_rad) / 360.

        # Precompute radial bin indices (k)
        y, x = np.indices(self.map_shape)
        self.radial_bins = np.sqrt((x - self.center)**2 + (y - self.center)**2).astype(np.int32)
    
    def compute_power_spectrum(self, image):
        """
        Compute the radially-binned 1-D power spectrum of a 2-D image.

        Parameters
        ----------
        image : ndarray, shape (N, N)
            Real-valued input map.

        Returns
        -------
        ell : ndarray
            Multipole moment values for each radial bin.
        power_spectrum_1d : ndarray
            Azimuthally averaged power spectrum in units of rad².
        """
        # FFT using pyFFTW
        fft_object = pyfftw.builders.fft2(image, threads=8)
        fft_map = fft_object()
        
        # Normalize the FFT by N^2
        fft_map_normalized = fft_map / (self.N**2)
        
        # Shift the zero frequency to the center
        data_ft_shifted = jnp.fft.fftshift(np.abs(fft_map_normalized))
        
        # Compute the 2D power spectrum
        power_spectrum_2d = (data_ft_shifted**2)

        # Radial profile using precomputed bins
        tbin = jnp.bincount(self.radial_bins.ravel(), power_spectrum_2d.ravel())
        nr = jnp.bincount(self.radial_bins.ravel())
        radial_profile = jnp.where(nr > 0, tbin / nr, 0)

        # Nyquist limit
        nyquist = self.N // 2
        powerspectrum = radial_profile[:nyquist]

        # 1D Power spectrum in [rad^2]
        power_spectrum_1d = powerspectrum * (self.pix_to_rad**2)

        # Multipole moments (ell)
        ks = jnp.arange(power_spectrum_1d.shape[0])
        # ell = 2. * np.pi * ks / (self.pix_to_rad * self.N)
        ell = 4. * np.pi * np.pi * ks 
        return ell, power_spectrum_1d

    
    def adjust_map_cls(self, input_map, target_cls, target_ells):
        """
        Adjust ``input_map`` so its power spectrum matches ``target_cls``.

        Scales Fourier amplitudes by ``sqrt(C_ell_target / C_ell_input)``
        while preserving Fourier phases.

        Parameters
        ----------
        input_map : ndarray, shape (N, N)
            Real-valued input map.
        target_cls : ndarray
            Target power spectrum values.
        target_ells : ndarray
            Multipole moments corresponding to ``target_cls``.

        Returns
        -------
        adjusted_map : ndarray, shape (N, N)
            Map with rescaled Fourier amplitudes.
        """
        # Ensure the input map has zero mean
        # input_map = input_map - np.mean(input_map)

        # Compute the 2D Fourier grid (if not precomputed)
        # if not hasattr(self, 'ell2d'):
        ell_x = 2 * np.pi * np.fft.fftfreq(input_map.shape[0], d=self.pix_to_rad)
        ell_y = 2 * np.pi * np.fft.fftfreq(input_map.shape[1], d=self.pix_to_rad)
        ell_x, ell_y = np.meshgrid(ell_x, ell_y)
        ell2d = np.sqrt(ell_x**2 + ell_y**2)

        # Interpolate target Cl spectrum onto 2D grid
        cl_interp = interp1d(target_ells, target_cls, bounds_error=False, fill_value=target_cls[-1])
        CLTarget2d = cl_interp(ell2d)

        # Compute input map's Cl and interpolate
        ell_input, power_spectrum_1d_input = self.compute_power_spectrum(input_map)
        cl_interp_input = interp1d(ell_input, power_spectrum_1d_input, bounds_error=False, fill_value=power_spectrum_1d_input[-1])
        CLInput2d = cl_interp_input(ell2d)

        # Compute scaling factors
        epsilon = 1e-25
        scaling_log = 0.5 * (np.log(CLTarget2d) - np.log(CLInput2d + epsilon))
        scaling = np.exp(scaling_log)

        # FFT the input map using pyFFTW
        fft_object = pyfftw.builders.fft2(input_map, threads=8)
        fft_map = fft_object()
        current_amplitudes = np.abs(fft_map)
        phases = np.angle(fft_map)

        # Scale Fourier amplitudes to match the target Cl
        adjusted_amplitudes = current_amplitudes * scaling
        adjusted_fft_map = adjusted_amplitudes  * np.exp(1j * phases)

        # IFFT to get the adjusted map
        ifft_object = pyfftw.builders.ifft2(adjusted_fft_map, threads=8)
        adjusted_map = ifft_object().real

        return adjusted_map

class PowerSpectrum:
    """
    Flat-sky angular power spectrum computation and Cls-based field adjustment.

    Computes the binned angular power spectrum of a 2-D convergence map using
    ``rfft2`` and adjusts Fourier amplitudes to match a target Cℓ curve while
    preserving Fourier phases (and hence non-Gaussian structure).

    Parameters
    ----------
    map : ndarray, shape (N, N)
        Reference map used to set the ℓ grid and bin edges.
    pixelsize : float
        Pixel size in degrees.

    Attributes
    ----------
    ell_min, ell_max : float
        Minimum and maximum multipole moments for the map.
    nbinsell : int
        Number of ℓ bins.
    l : ndarray, shape (N, N//2+1)
        2-D multipole-moment grid matching the ``rfft2`` output layout.
    ell_edges : ndarray, shape (nbinsell+1,)
        Bin edges for the power spectrum.
    """

    def __init__(self, map, pixelsize):
        """
        Parameters
        ----------
        map : ndarray, shape (N, N)
            Reference map used to initialise the ℓ grid.
        pixelsize : float
            Pixel size in degrees.
        """
        self.pixelsize = pixelsize
        self.map_size = map.shape[0]
        self.ell_min = 180. / (self.map_size * np.deg2rad(self.pixelsize))
        self.ell_max = 90 / np.deg2rad(self.pixelsize)
        self.deltaell = 200
        self.nbinsell = int((self.ell_max - self.ell_min) / self.deltaell)
        self.pixel_size_rad_perpixel = np.pi * self.pixelsize / (180. * 60.)
        self.lpix = 2. * np.pi / self.pixel_size_rad_perpixel / 360.
        self.lx = rfftfreq(self.map_size) * self.map_size * self.lpix
        self.ly = fftfreq(self.map_size) * self.map_size * self.lpix
        self.l = jnp.sqrt(self.lx[np.newaxis, :] ** 2 + self.ly[:, np.newaxis] ** 2)
        self.ell_edges = jnp.linspace(self.ell_min, self.ell_max, num=self.nbinsell + 1)

    def calculate_Cls(self, map):
        """
        Calculate the binned angular power spectrum of a map.

        Parameters
        ----------
        map : ndarray, shape (N, N)
            Real-valued input map.

        Returns
        -------
        ell_edges : ndarray, shape (nbinsell+1,)
            Bin edges in multipole moment space.
        ell_bins : ndarray, shape (nbinsell,)
            Bin centres.
        cls_values : ndarray, shape (nbinsell,)
            Azimuthally averaged power spectrum in units of rad⁴.
        """
        map_ft = rfft2(map)
        power_spectrum = jnp.abs(map_ft) ** 2

        # Digitize frequencies into bins
        bin_idx = jnp.digitize(self.l.ravel(), self.ell_edges) - 1
        valid_mask = (bin_idx >= 0) & (bin_idx < self.nbinsell)

        # Aggregate power and counts per bin
        power_l = jnp.zeros(self.nbinsell)
        hits = jnp.zeros(self.nbinsell)
        power_l = power_l.at[bin_idx[valid_mask]].add(power_spectrum.ravel()[valid_mask])
        hits = hits.at[bin_idx[valid_mask]].add(1)

        # Compute Cls
        cls_values = jnp.where(hits > 0, power_l / hits, 0.0)
        ell_bins = 0.5 * (self.ell_edges[1:] + self.ell_edges[:-1])
        normalization = (jnp.deg2rad(self.pixelsize * self.map_size) / self.map_size ** 2) ** 2
        return self.ell_edges, ell_bins, cls_values * normalization

    def generate_field_with_target_cls(self, input_field, target_cls, target_ells):
        """
        Rescale Fourier amplitudes of ``input_field`` to match ``target_cls``.

        Preserves Fourier phases so that non-Gaussian structure is retained.
        Below-range ℓ modes use the large-scale (first) target Cℓ value;
        above-range modes use the small-scale (last) value.

        Parameters
        ----------
        input_field : ndarray, shape (N, N)
            Real-valued input map.
        target_cls : ndarray
            Target power spectrum values.
        target_ells : ndarray
            Multipole moments corresponding to ``target_cls``.

        Returns
        -------
        ndarray, shape (N, N)
            Adjusted field with power spectrum matching ``target_cls``.

        Raises
        ------
        AssertionError
            If ``len(target_cls) != len(target_ells)`` or any element of
            ``target_cls`` is negative.
        """
        assert len(target_cls) == len(target_ells), "target_cls and target_ells must have the same length"
        assert jnp.all(target_cls >= 0), "All target_cls values must be non-negative"

        field_ft = rfft2(input_field)
        _, ell_bins, field_cls = self.calculate_Cls(input_field)

        # Interpolate Cls
        # fill_value is a 2-tuple (below_range, above_range) so that low-ell extrapolation
        # uses the large-scale (first) value rather than the high-ell tail.
        field_cls_interp = interp1d(ell_bins, field_cls, kind="linear", bounds_error=False,
                                    fill_value=(field_cls[0], field_cls[-1]))
        target_cls_interp = interp1d(target_ells, target_cls, kind="linear", bounds_error=False,
                                     fill_value=(target_cls[0], target_cls[-1]))

        Cl_field = field_cls_interp(self.l)
        Cl_target = target_cls_interp(self.l)

        # Adjust the amplitude
        adjustment_factor = jnp.sqrt(Cl_target / (Cl_field + 1e-20))
        adjusted_amplitude = jnp.abs(field_ft) * adjustment_factor

        # Preserve phase and inverse transform
        adjusted_field_ft = adjusted_amplitude * jnp.exp(1j * jnp.angle(field_ft))
        return irfft2(adjusted_field_ft).real
    
# if __name__ == "__main__":
#     # Generate a random input map
#     input_map = np.random.rand(256, 256)
#     pixelsize = 0.5  # Example pixel size in degrees

#     # Instantiate the PowerSpectrum class
#     power_spectrum_calculator = PowerSpectrum(input_map, pixelsize)

#     # Compute the power spectrum
#     ell_edges, ell_bins, cls_values = power_spectrum_calculator.calculate_Cls(input_map)
#     print("Power spectrum calculated:", cls_values)

#     # Generate a field with a target power spectrum
#     target_ells = np.linspace(ell_edges.min(), ell_edges.max(), len(cls_values))
#     target_cls = cls_values * 1.2  # Example modification of Cls values
#     generated_field = power_spectrum_calculator.generate_field_with_target_cls(input_map, target_cls, target_ells)
#     print("Generated field with target power spectrum.")


def fourier_coordinate(x, y, map_size):
    """
    Compute the flat index into a row-major ``rfft2`` output array.

    Parameters
    ----------
    x : int
        Row index.
    y : int
        Column index.
    map_size : int
        Width of the map (number of columns).

    Returns
    -------
    int
        Flat index ``(map_size // 2 + 1) * x + y``.
    """
    return (map_size // 2 + 1) * x + y

def calculate_Cls_(map, angle, ell_min, ell_max, n_bins):
    """
    Calculate the angular power spectrum of a map using JAX ``lax.fori_loop``.

    Parameters
    ----------
    map : ndarray, shape (N, N)
        Input convergence map.
    angle : float
        Field-of-view side length in degrees.
    ell_min : float
        Minimum multipole moment.
    ell_max : float
        Maximum multipole moment.
    n_bins : int
        Number of ℓ bins.

    Returns
    -------
    ell_edges : ndarray, shape (n_bins+1,)
        Bin edges.
    ell_bins : ndarray, shape (n_bins,)
        Bin centres.
    cls_values : ndarray, shape (n_bins,)
        Normalised angular power spectrum.
    """
    ell_min = jnp.array(ell_min)
    ell_max = jnp.array(ell_max)
    # n_bins = jnp.array(n_bins, int)

    # Calculate the Fourier Transforms
    map_ft = rfft2(map) ## rfft2
    map_ft = map_ft.flatten()
    ell_edges = jnp.linspace(ell_min, ell_max, num=n_bins+1)

    # Define pixel physical size in Fourier space
    pixel_size = angle*60 / map.shape[0]
    pixel_size_rad_perpixel = np.pi * pixel_size / 180. / 60.
    lpix = 2. * np.pi / pixel_size_rad_perpixel / 360. 
    # Initialize arrays to store power and hits for each ell bin
    power_l = jnp.zeros(n_bins)
    hits = jnp.zeros(n_bins)
    
    def loop_body(j, val):
        i, power_l, hits = val
        lx = jnp.minimum(i, map.shape[1] - i) * lpix
        ly = j * lpix
        l = jnp.sqrt(lx**2. + ly**2.)
        pixid = fourier_coordinate(i, j, map.shape[1])
        bin_idx = jnp.digitize(l, ell_edges) - 1
        power_l = power_l.at[bin_idx].add(jnp.abs(map_ft[pixid]**2.))
        hits = hits.at[bin_idx].add(1)
        return i, power_l, hits

    def outer_loop_body(i, val):
        _, power_l, hits = val
        _, power_l, hits = jax.lax.fori_loop(0, map.shape[0], loop_body, (i, power_l, hits))
        return i, power_l, hits

    _, power_l, hits = jax.lax.fori_loop(0, map.shape[1], outer_loop_body, (0, power_l, hits))

    # Calculate Cls based on the accumulated power and hits
    cls_values = jnp.where(hits > 0, power_l / hits, 0.0)  # Ensure no division by zero
    # cls_values = power_l/hits
    cls_values = jnp.maximum(cls_values, 0)  # Clip negative values to zero if any
    ell_bins = 0.5 * (ell_edges[1:] + ell_edges[:-1])
    normalization = (jnp.deg2rad(angle) / (map.shape[0] * map.shape[0]))**2

    return jnp.array(ell_edges), jnp.array(ell_bins), jnp.array(cls_values * normalization)

# @partial(jax.jit, static_argnums=(1, 4, 5))
def generate_field_with_target_cls_(input_field, angle, target_cls, target_ells, ell_max=40000, n_bins=50):
    """
    Adjust the power spectrum of ``input_field`` to match ``target_cls``.

    Standalone (non-method) version that accepts explicit field-of-view and
    ℓ-range arguments.  Calls :func:`calculate_Cls` internally and uses the
    same two-sided ``fill_value`` interpolation as
    :meth:`PowerSpectrum.generate_field_with_target_cls`.

    Parameters
    ----------
    input_field : jnp.ndarray, shape (N, N)
        Real-valued 2-D input map.
    angle : float
        Field-of-view side length in degrees.
    target_cls : jnp.ndarray
        Target power spectrum values.
    target_ells : jnp.ndarray
        Multipole moments corresponding to ``target_cls``.
    ell_max : int, optional
        Maximum multipole moment.  Default is ``40000``.
    n_bins : int, optional
        Number of ℓ bins for the power spectrum calculation.  Default is ``50``.

    Returns
    -------
    jnp.ndarray, shape (N, N)
        Adjusted field whose power spectrum matches ``target_cls``.

    Raises
    ------
    AssertionError
        If ``len(target_cls) != len(target_ells)`` or any element of
        ``target_cls`` is negative.
    """
    shape = input_field.shape
    assert len(target_cls) == len(target_ells), "target_cls and target_ells must have the same length"
    assert jnp.all(target_cls >= 0), "All target_cls values must be non-negative"
    
    # Fourier transform of the input field
    field_ft = rfft2(input_field) ##
    
    # Calculate the Cls for the input field
    ell_edges, ell_bins, field_cls = calculate_Cls(input_field, angle, 0, ell_max, n_bins)
    map = input_field
    # Compute lpix and l values for FFT pixels
    pixel_size = angle*60 / map.shape[0]
    pixel_size_rad_perpixel = np.pi * pixel_size / 180. / 60.
    lpix = 2. * np.pi / pixel_size_rad_perpixel / 360. 
    lx = rfftfreq(shape[0]) * shape[0] * lpix ##
    ly = fftfreq(shape[1]) *  shape[1] * lpix
    l = jnp.sqrt(lx[np.newaxis, :]**2 + ly[:, np.newaxis]**2)
    
    # Interpolate Cls for the input field and the target
    field_cls_interp = interp1d(ell_bins, field_cls, kind="linear", bounds_error=False,
                                fill_value=(field_cls[0], field_cls[-1]))
    Cl_field = field_cls_interp(l)
    target_cls_interp = interp1d(target_ells, target_cls, kind="linear", bounds_error=False,
                                 fill_value=(target_cls[0], target_cls[-1]))
    Cl_target = target_cls_interp(l)
    
    # Adjust the amplitude based on the target Cls
    adjustment_factor = jnp.sqrt(Cl_target / Cl_field)
    adjusted_amplitude = jnp.abs(field_ft) * adjustment_factor
    
    # Recombine adjusted amplitude with original phase
    adjusted_field_ft = adjusted_amplitude * jnp.exp(1j * jnp.angle(field_ft))
    # print(adjusted_field_ft.shape) 
    # Inverse Fourier Transform to get the adjusted field in real space
    adjusted_field = irfft2(adjusted_field_ft) ##
    # print(shape, adjusted_field.shape, field_ft.shape)
    return adjusted_field.real