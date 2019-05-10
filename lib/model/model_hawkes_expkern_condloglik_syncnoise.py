from math import exp, log
import numpy as np
from numba import njit, jit
from scipy.optimize import approx_fprime

from tick.hawkes.model import ModelHawkesExpKernLogLik

from ._heavy import _heavy
from .model_hawkes_syncnoise import (
    ModelHawkesCondLogLikSyncNoise, enforce_fitted
)
from .model_hawkes_smoothedexpkern import (
    SyncNoiseModelHawkesSmoothSigmoidExpKern
)


SMOOTH_MODEL_APPROX_CUTOFF = 1e-3


@jit(nopython=True, fastmath=True)
def _loglikelihood(baseline, adj, decay, decay_neg, gamma, dim, events,
                   end_time, min_indices, max_indices, min_indices_integrated):
    value = 0.0
    # Loop over each dimension (can be parallelized)
    for i in range(dim):
        # Loop over events of dimension `i`
        for k, t_k_i in enumerate(events[i]):
            int_ik = 0.0
            int_ik += baseline[i]
            for j in range(dim):
                if adj[i, j] == 0:
                    continue
                int_ikj = 0.0
                # For all events t_l_j s.t.: t_k_i - cutoff < t_l_j < t_k_i
                min_ijk = min_indices[i][j][k]
                max_ijk = max_indices[i][j][k]
                for t_l_j in events[j][min_ijk:max_ijk]:
                    t = t_k_i - t_l_j
                    int_ikj += exp(-decay * t)
                int_ik += adj[i, j] * int_ikj
            value += log(int_ik)
        # Integrated intensity part
        intint_i = 0.0
        for t_k_i in events[i][min_indices_integrated[i]:]:
            t = end_time - t_k_i
            intint_i += (1 - exp(-decay * t)) / decay
        value -= adj[:, i].sum() * intint_i
    value -= baseline.sum() * end_time
    return value


def loglikelihood(baseline, adj, decay, decay_neg, gamma, dim, events,
                  end_time, approx):
    min_indices, max_indices = _heavy.precompute_batch_indices(
        events, 1e-5, decay)
    min_indices = tuple(map(tuple, map(np.array, min_indices)))
    max_indices = tuple(map(tuple, map(np.array, max_indices)))
    cutoff = -np.log(approx) / max(decay, gamma - decay_neg)
    min_indices_integrated = np.array([np.searchsorted(
        events[i], end_time - cutoff) for i in range(dim)])
    return _loglikelihood(baseline, adj, decay, decay_neg, gamma, dim,
                          tuple(events), end_time, min_indices, max_indices,
                          min_indices_integrated)


class ModelHawkesExpKernCondLogLikSyncNoise(ModelHawkesCondLogLikSyncNoise):
    """
    Model a multivariate Hawkes process with Exponential kernels and
    Synchronization noise using the negative log-likelihood loss function. This
    object provides an interface to compute the value of the loss as well as
    its gradient.

    The parameters `coeffs` of the model are (`noise_coeffs`, `hawkes_coeffs`),
    where `noise_coeffs` corresponds to the noise assignment in each dimension
    and `hawkes_coeffs` corresponds to the Hawkes process parameters with first
    the baseline in each dimension, then the kernel weights (same convention as
    in the library `tick`).
    """

    ALLOWED_APPROX_TYPE = set(['findiff', 'smooth'])

    def __init__(self, decay, n_threads=1, approx_type='smooth', epsilon=1e-3,
                 decay_neg=1000, gamma=5000):
        super().__init__(decay, n_threads)
        self._model_init(approx_type, epsilon, decay_neg, gamma)

    def _model_init(self, approx_type, epsilon, decay_neg, gamma):
        # Init the model for the Hawkes parameters
        self.hawkes_model = ModelHawkesExpKernLogLik(
            self.decay, self.n_threads)
        # Set the type of approximation
        self.approx_type = approx_type
        if self.approx_type == 'findiff':
            # Epsilon used for the fin.-diff. approx. of the noise gradient
            self.epsilon = epsilon
            if self.epsilon <= 0:
                raise ValueError('`epsilon` must be positive')
        elif self.approx_type == 'smooth':
            # Exponential Decay in negative time, used for the smooth appprox
            # of the noise gradient
            if decay_neg <= 0:
                raise ValueError('`decay_neg` must be positive')
            # Rate of transition between negative and positive exponenetial
            # rate used for the smooth appprox of the noise gradient
            if gamma <= 0:
                raise ValueError('`gamma` must be positive')
            self.smooth_noise_model = SyncNoiseModelHawkesSmoothSigmoidExpKern(
                decay=self.decay, decay_neg=decay_neg, gamma=gamma,
                approx=SMOOTH_MODEL_APPROX_CUTOFF)
        else:
            raise ValueError(
                '`approx_type` is `{:s}`, but must be in: {:s}'.format(
                    str(approx_type), ', '.join(ALLOWED_APPROX_TYPE)))

    def _fit_hawkes_model(self, events, end_time):
        try:
            self.hawkes_model.fit(events, end_time)
        except RuntimeError:
            # Fix tick bug in version 0.4.0.0
            self.hawkes_model = ModelHawkesExpKernLogLik(self.decay,
                                                         self.n_threads)
            self.hawkes_model.fit(events, end_time)

    @enforce_fitted
    def _loss_noise(self, noise_coeffs, hawkes_coeffs):
        """
        Internal function used to approximate the noise gradient. Returns the
        value of the loss function at `hawkes_coeffs` with noise assignment
        `noise_coeffs`.
        """
        # Condition the data on the given noise assignment
        self._condition_events_on_noise(noise_coeffs)
        # Compute the loss
        loss = self._loss(hawkes_coeffs)
        return loss

    @enforce_fitted
    def loss(self, coeffs):
        """
        Evaluate the negative log-likelihood at parameters `coeffs`.
        """
        noise_coeffs, hawkes_coeffs = self._split_coeffs(coeffs)
        # Condition the data on the given noise assignment
        self._condition_events_on_noise(noise_coeffs)
        try:
            # Compute the loss
            loss = self._loss(hawkes_coeffs)
        except RuntimeError:
            # In case some parameters are negative
            return np.inf
        return loss

    @enforce_fitted
    def grad_noise(self, coeffs):
        """
        Return the value of the noise gradient at `coeffs`.
        """
        noise_coeffs, hawkes_coeffs = self._split_coeffs(coeffs)
        if self.approx_type == 'findiff':
            return approx_fprime(
                noise_coeffs,  # xk
                self._loss_noise,  # f
                self.epsilon,  # epsilon
                hawkes_coeffs  # *args passed to `f`
            )
        elif self.approx_type == 'smooth':
            self._condition_events_on_noise(noise_coeffs)
            self.smooth_noise_model.fit(self.fitted_events, self.fitted_end_time)
            return self.smooth_noise_model.grad(hawkes_coeffs)

    @enforce_fitted
    def grad_hawkes(self, coeffs):
        """
        Return the value of the Hawkes gradient at `coeffs`.
        """
        noise_coeffs, hawkes_coeffs = self._split_coeffs(coeffs)
        self._condition_events_on_noise(noise_coeffs)
        hawkes_grad = self.hawkes_model.grad(hawkes_coeffs)
        return hawkes_grad

    @enforce_fitted
    def grad(self, coeffs):
        """
        Compute the gradient of the negative log-likelihood at parameters
        `coeffs`.
        """
        return np.hstack((self.grad_noise(coeffs), self.grad_hawkes(coeffs)))
