from math import exp, log
import numpy as np
from numba import njit, jit
from scipy.optimize import approx_fprime

from tick.hawkes.model import ModelHawkesExpKernLogLik

from ._heavy import _heavy
from .model_hawkes_smoothedexpkern import SyncNoiseModelHawkesSmoothSigmoidExpKern


def enforce_fitted(func):
    """
    Enforce that the `_fitted` attribute of the object is set to `True`
    """
    def wrapper(*args):
        if not args[0]._fitted:
            raise ValueError('call ``fit`` before using ``{:s}``'.format(
                func.__name__))
        return func(*args)
    return wrapper


class ModelHawkesCondLogLikSyncNoise(object):
    """
    Abstract model a multivariate Hawkes process under Synchronization noise
    for a generic loss function. The model provides an interface to compute
    the value of the loss as well as its gradient.

    The parameters `coeffs` of the model are (`noise_coeffs`, `hawkes_coeffs`),
    where `noise_coeffs` corresponds to the noise assignment in each dimension
    and `hawkes_coeffs` corresponds to the Hawkes process parameters with first
    the baseline in each dimension, then the kernel weights (same convention as
    in the library `tick`).
    """

    def __init__(self, decay, n_threads=1):
        # Some attributes are set at fitting time
        self._fitted = False
        if decay <= 0:
            raise ValueError('`decay` must be positive')
        self.decay = decay
        # Number of theads for the `hawkes_model` object
        self.n_threads = n_threads

    def _fit_hawkes_model(self, events, end_time):
        raise NotImplementedError('Must be instanciated')

    def fit(self, events, end_time):
        """
        Fit the model to the noisy observed timestamps.
        """
        self._fitted = True
        # Set the number of parameters
        self.dim = len(events)
        self.n_params = self.dim + self.dim * (self.dim + 1)
        # Set the observed data (must be a single realization)
        if not isinstance(events[0], np.ndarray):
            raise TypeError("`events` must be a list of `np.ndarray`")
        self.obs_timestamps = events
        # Set the observed end_time
        if not isinstance(end_time, (int, float)):
            raise TypeError("`end_time` must be a single number")
        self.obs_end_time = end_time
        # Fit the Hawkes model to this data (must be a single realization)
        self._fit_hawkes_model(events, end_time)
        # Set some attributes
        self.fitted_events = events
        self.fitted_end_time = end_time
        # Current noise assignment
        self.noise_val = np.zeros(self.dim)

    def _split_coeffs(self, coeffs):
        """
        Separate the noise and Hawkes parameters from the vector `coeffs`.
        """
        noise_coeffs = coeffs[:self.dim]
        hawkes_coeffs = coeffs[self.dim:]
        return noise_coeffs, hawkes_coeffs

    @enforce_fitted
    def _condition_events_on_noise(self, z_per_dim):
        """
        Remove the delays `z_sample_per_dim` from each dimension and adjust the
        observations window accordingly.
        """
        # Shift the start time of this sample by the minimum delay value
        new_start_time = 0.0 - min(z_per_dim)
        # Remove the maximum delay value to the end time of the sample
        new_end_time = self.obs_end_time - max(z_per_dim) - new_start_time
        # Remove these
        new_events = list()
        for m, events_m in enumerate(self.obs_timestamps):
            shift_m = z_per_dim[m] + new_start_time
            new_events_m = events_m - shift_m
            min_idx, max_idx = np.searchsorted(new_events_m, [0.0, new_end_time])
            new_events_m = new_events_m[min_idx:max_idx]
            new_events.append(new_events_m)
        # Fit the Hawkes model to this data
        self._fit_hawkes_model(new_events, new_end_time)
        # Set some attributes
        self.fitted_events = new_events
        self.fitted_end_time = new_end_time
        self.noise_val = z_per_dim.copy()

    @enforce_fitted
    def _loss(self, hawkes_coeffs):
        """
        Compute the value of the loss function at `hawkes_coeffs` for a fixed
        already-fitted noise assignment. This function is used to avoid fit the
        same noise assignment multiple times. Use `loss` instead to fit the
        noise assignment in addition to the hawkes coefficients.
        """
        return self.hawkes_model.loss(hawkes_coeffs)

    @enforce_fitted
    def loss(self, coeffs):
        """
        Evaluate the loss function at parameters `coeffs`.
        """
        raise NotImplementedError('Must be instanciated')

    @enforce_fitted
    def grad(self, coeffs):
        """
        Compute the gradient of the loss function at parameters `coeffs`.
        """
        raise NotImplementedError('Must be instanciated')
