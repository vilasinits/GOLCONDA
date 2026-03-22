import numpy as np
import scipy.special
import jax
import jax.numpy as jnp
from jax.numpy.fft import fftfreq
from functools import partial
from typing import List
from lenspack.image.transforms import starlet2d


# ---------------------------------------------------------------------------
# JIT-compiled scan kernel (no scipy inside — pure JAX FFTs)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=())
def _apply_filters_scan(
    field_ft: jnp.ndarray,
    filters: jnp.ndarray,
) -> jnp.ndarray:
    """
    Apply a pre-computed stack of top-hat filters via ``lax.scan``.

    The entire multi-scale loop — all ``irfft2`` calls and difference steps —
    compiles into a single XLA program, eliminating Python overhead per scale.

    Parameters
    ----------
    field_ft : (N, N//2+1) complex array
        ``rfft2`` of the input map.
    filters : (num_scales, N, N//2+1) float array
        Fourier-space top-hat filters for each scale, pre-computed with
        ``scipy.special.j1`` at full float64 accuracy.

    Returns
    -------
    jnp.ndarray, shape (num_scales+1, N, N)
        ``[w_1, …, w_nscales, coarse]``
    """
    def step(coarse, filt):
        new_coarse = jnp.fft.irfft2(field_ft * filt)
        wavelet = coarse - new_coarse
        return new_coarse, wavelet

    init_coarse = jnp.fft.irfft2(field_ft)
    final_coarse, wavelet_stack = jax.lax.scan(step, init_coarse, filters)
    return jnp.concatenate([wavelet_stack, final_coarse[None]], axis=0)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class WaveletDecomposer:
    """
    Multi-scale wavelet decomposer (top-hat or starlet).

    Top-hat optimisations
    ---------------------
    * ``scipy.special.j1`` is called **once** per unique (shape, num_scales)
      pair, at full float64 accuracy, to compute all scale filters up front.
    * The filters are cached; repeated calls with the same map shape and
      number of scales skip the Bessel-function step entirely.
    * ``rfft2`` / ``irfft2`` (half-spectrum) replace the full complex FFT,
      halving memory and compute for real-valued maps.
    * The scale loop is compiled into a single XLA program via
      ``jax.lax.scan`` — no Python overhead per scale.

    Parameters
    ----------
    num_scales : int
        Default number of decomposition scales.
    """

    def __init__(self, num_scales: int = 3) -> None:
        self.num_scales = num_scales
        self._filter_cache: dict = {}   # keyed by (image_shape, num_scales)

    # ------------------------------------------------------------------
    # Grid setup
    # ------------------------------------------------------------------

    def set_size_dependent_params(self, image_shape, L=None):
        """
        Precompute the |k| grid for a given image shape.

        Called automatically by ``decompose`` when ``recalculate_params=True``.

        Parameters
        ----------
        image_shape : tuple of int
        L : float, optional
            Physical length of one side. Defaults to ``image_shape[0]``.
        """
        self.image_shape = image_shape
        self.L = L or image_shape[0]
        self.dx = self.L / image_shape[0]

        # rfft2 output has shape (N, N//2+1):
        #   axis-0 (rows) → fftfreq  (N values)
        #   axis-1 (cols) → rfftfreq (N//2+1 values)
        krows = fftfreq(image_shape[0], self.dx)
        kcols = jnp.fft.rfftfreq(image_shape[1], self.dx)
        kr_grid, kc_grid = jnp.meshgrid(krows, kcols, indexing='ij')
        self.k_mag = 2.0 * jnp.pi * jnp.sqrt(kr_grid ** 2 + kc_grid ** 2)

    # ------------------------------------------------------------------
    # Filter helpers
    # ------------------------------------------------------------------

    def _compute_filters(self, num_scales: int) -> jnp.ndarray:
        """
        Compute and cache the (num_scales, N, N//2+1) filter stack.

        Uses ``scipy.special.j1`` for float64 accuracy.  The result is
        cached so repeated calls with the same arguments are free.
        """
        key = (self.image_shape, num_scales)
        if key not in self._filter_cache:
            scale_exps = 2.0 ** np.arange(1, num_scales + 1)          # (S,)
            k_np = np.asarray(self.k_mag)                               # (N, N//2+1)
            u_all = k_np[None, :, :] * scale_exps[:, None, None]       # (S, N, N//2+1)
            u_safe = np.where(u_all == 0.0, 1e-7, u_all)
            filters = 2.0 * scipy.special.j1(u_safe) / u_safe          # (S, N, N//2+1)
            self._filter_cache[key] = jnp.asarray(filters)
        return self._filter_cache[key]

    def get_top_hat_filter(self, window_radius: float) -> jnp.ndarray:
        """
        Return the top-hat Fourier filter for a single radius (public API).

        Parameters
        ----------
        window_radius : float
            Top-hat radius in pixel units.
        """
        k_np = np.asarray(self.k_mag)
        u = k_np * window_radius
        u_safe = np.where(u == 0.0, 1e-7, u)
        return jnp.asarray(2.0 * scipy.special.j1(u_safe) / u_safe)

    def get_th_smooth_map(self, map_in: np.ndarray, window_radius: float) -> np.ndarray:
        """
        Smooth ``map_in`` with the top-hat filter of ``window_radius``.

        Parameters
        ----------
        map_in : ndarray, shape (N, M)
        window_radius : float
            Radius in pixel units.

        Returns
        -------
        np.ndarray, same shape as ``map_in``.
        """
        self.set_size_dependent_params(map_in.shape)
        field_ft = jnp.fft.rfft2(jnp.asarray(map_in))
        filt = self.get_top_hat_filter(window_radius)
        return np.asarray(jnp.fft.irfft2(field_ft * filt))

    # ------------------------------------------------------------------
    # Decomposition
    # ------------------------------------------------------------------

    def decompose_with_tophat(self) -> np.ndarray:
        """
        Top-hat wavelet decomposition of ``self.input_map``.

        Filters are computed/retrieved from cache, then the FFT scan is
        run under JIT.

        Returns
        -------
        np.ndarray, shape (num_scales+1, N, M)
        """
        filters = self._compute_filters(self.num_scales)
        field_ft = jnp.fft.rfft2(jnp.asarray(self.input_map))
        return np.array(_apply_filters_scan(field_ft, filters))

    def decompose_with_starlet(self) -> np.ndarray:
        """Starlet (à trous) wavelet decomposition via lenspack."""
        return starlet2d(self.input_map, self.num_scales)

    def reconstruct(self, coefficients: List[np.ndarray]) -> np.ndarray:
        """Reconstruct map from wavelet coefficients by summing all scales."""
        return np.sum(coefficients, axis=0)

    def decompose(
        self,
        input_map: np.ndarray,
        num_scales: int = 1,
        filter_type: str = 'tophat',
        recalculate_params: bool = True,
    ) -> np.ndarray:
        """
        Decompose ``input_map`` into wavelet scales.

        Parameters
        ----------
        input_map : ndarray
            2-D real-valued input map.
        num_scales : int
            Number of wavelet scales (coarse residual is always appended).
        filter_type : {'tophat', 'starlet'}
        recalculate_params : bool
            Re-derive the frequency grid from the map shape.  Set ``False``
            only when calling repeatedly with identically-shaped maps.

        Returns
        -------
        np.ndarray, shape (num_scales+1, N, M)
            ``[w_1, w_2, …, w_nscales, coarse]``
        """
        if num_scales is not None:
            self.num_scales = num_scales

        if recalculate_params:
            self.set_size_dependent_params(input_map.shape)

        self.input_map = input_map

        if filter_type == 'tophat':
            return self.decompose_with_tophat()
        elif filter_type == 'starlet':
            return self.decompose_with_starlet()
        else:
            raise ValueError(
                f"Unknown filter type: {filter_type!r}. "
                "Supported values: 'tophat', 'starlet'."
            )


if __name__ == "__main__":
    input_image = np.random.rand(256, 256)
    d = WaveletDecomposer()
    coeffs = d.decompose(input_image, num_scales=5, filter_type='tophat')
    print("Shape:", coeffs.shape)
    print("Reconstruction error:", np.max(np.abs(coeffs.sum(axis=0) - input_image)))
