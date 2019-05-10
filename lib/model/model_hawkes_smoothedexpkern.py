from math import exp, log
from numba import njit
import numpy as np

from ._heavy._heavy import (
    precompute_batch_min_indices,
    precompute_batch_max_indices
)


@njit(fastmath=True)
def _loglikelihood(baseline, adj, decay, decay_neg, gamma, dim, events, end_time,
                   min_indices, max_indices, min_indices_integrated):
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
                # For all events t_l_j s.t.: t_k_i - cutoff < t_l_j < t_k_i
                for t_l_j in events[j][min_indices[i][j][k]:max_indices[i][j][k]]:
                    t = t_k_i - t_l_j
                    b = max(-decay * t, -(gamma - decay_neg) * t, -gamma * t)
                    int_ik += adj[i, j] * (
                        exp(-decay * t - b) + exp(-(gamma - decay_neg) * t - b)
                    ) / (
                        exp(-b) + exp(-gamma * t - b)
                    )
            value += log(int_ik)
        # Integrated intensity part
        intint_i = 0.0
        for t_k_i in events[i][min_indices_integrated[i]:]:
            t = end_time - t_k_i
            intint_i += (1 - exp(-decay * t)) / decay
            intint_i += (log(2) - log(1 + exp(-(gamma - decay_neg) * t))) / (gamma - decay_neg)
        value -= adj[:, i].sum() * intint_i
    value -= baseline.sum() * end_time
    return value


@njit(fastmath=True)
def _kernel(t, decay, decay_neg, gamma):
    b = max(-decay * t, -(gamma - decay_neg) * t, -gamma * t)
    return (
        exp(-decay * t - b) + exp(-(gamma - decay_neg) * t - b)
    ) / (
        exp(-b) + exp(-gamma * t - b)
    )


@njit(fastmath=True)
def _kernel_prime(t, decay, decay_neg, gamma):
    b = max(
        -decay * t,
        -(decay + gamma) * t,
        -(gamma - decay_neg) * t,
        -(2 * gamma - decay_neg) * t,
        -gamma * t,
        -2 * gamma * t
    )
    num = (
        - decay * exp(-decay * t - b)
        - (decay - gamma) * exp(-(decay + gamma) * t - b)
        - (gamma - decay_neg) * exp(-(gamma - decay_neg) * t - b)
        + decay_neg * exp(-(2 * gamma - decay_neg) * t - b)
    )
    denom = (
        exp(-b)
        + 2 * exp(-gamma * t - b)
        + exp(-2 * gamma * t - b)
    )
    return num / denom


@njit(fastmath=True)
def _int_kernel_prime(t, decay, decay_neg, gamma):
    return _kernel(t, decay, decay_neg, gamma)


@njit(fastmath=True)
def _numba_grad(baseline, adj, decay, decay_neg, gamma, dim, events, end_time,
                min_indices, max_indices, min_indices_integrated):
    grad = np.zeros(dim)
    # Compute the number of events
    num_events = 0
    for events_i in events:
        num_events += len(events_i)
    # Loop over each dimension (can be parallelized)
    for i in range(dim):
        grad_i = 0.0
        # Loop over events of dimension `i`
        for k, t_k_i in enumerate(events[i]):
            # Compute the numerator
            num = 0.0
            for j in range(dim):
                # dimension `i` should be excluded
                if (j == i) or (adj[i, j] <= 0.0):
                    continue
                # For all events t_l_j s.t.: t_k_i - cutoff < t_l_j < t_k_i
                for t_l_j in events[j][min_indices[i][j][k]:max_indices[i][j][k]]:
                    num -= adj[i, j] * _kernel_prime(
                        t_k_i - t_l_j, decay, decay_neg, gamma)
            # Compute the denominator
            if num != 0:
                denom = 0.0
                denom += baseline[i]
                for j in range(dim):
                    if (adj[i, j] <= 0.0):
                        continue
                    # For all events t_l_j s.t.: t_k_i - cutoff < t_l_j < t_k_i
                    for t_l_j in events[j][min_indices[i][j][k]:max_indices[i][j][k]]:
                        denom += adj[i, j] * _kernel(
                            t_k_i - t_l_j, decay, decay_neg, gamma)
                grad_i += num / denom
        # Loop over events of all other dimensions but `i`
        for j in range(dim):
            if (j == i) or (adj[j, i] <= 0.0):
                continue    
            for l, t_l_j in enumerate(events[j]):
                # Compute the numerator
                num = 0.0
                # For all events t_k_i s.t.: t_l_j - cutoff < t_k_i < t_l_j
                for t_k_i in events[i][min_indices[j][i][l]:max_indices[j][i][l]]:
                    num += adj[j, i] * _kernel_prime(
                        t_l_j - t_k_i, decay, decay_neg, gamma)
                if num != 0:
                    # Compute the denominator
                    denom = 0.0
                    denom += baseline[j]
                    for m in range(dim):
                        if (adj[j, m] <= 0.0):
                            continue
                        # For all events t_k_m such that:
                        # t_l_j - cutoff < t_k_m < t_l_j
                        for t_k_m in events[m][min_indices[j][m][l]:max_indices[j][m][l]]:
                            denom += adj[j, m] * _kernel(
                                t_l_j - t_k_m, decay, decay_neg, gamma)
                    grad_i += num / denom
        # Integrated intensity part
        last_term_i = 0.0
        for t_k_i in events[i][min_indices_integrated[i]:]:
            t = end_time - t_k_i
            last_term_i += _int_kernel_prime(t, decay, decay_neg, gamma)
        grad_i += adj[:, i].sum() * last_term_i
        # Negative and normalization (for tick objective)
        grad[i] = -grad_i / num_events
    return grad


class SyncNoiseModelHawkesSmoothSigmoidExpKern(object):

    def __init__(self, decay, decay_neg, gamma, approx=1e-5):
        self.decay = decay
        self.decay_neg = decay_neg
        self.gamma = gamma
        self.approx = approx

    def fit(self, events, end_time):
        self.events = tuple(events)
        self.end_time = end_time
        self.dim = len(self.events)
        self.num_events = sum(map(len, events))

        # Compute the min cutoff indices for the intensity
        min_indices = precompute_batch_min_indices(
            events, self.approx, self.decay)
        self.min_indices = tuple(map(tuple, map(np.array, min_indices)))
        # Compute the max cutoff indices for the intensity
        max_indices = precompute_batch_max_indices(
            events, self.approx, self.decay_neg)
        self.max_indices = tuple(map(tuple, map(np.array, max_indices)))
        # Compute the cutoff indices for the integrated intensity
        cutoff = -log(self.approx) / self.decay
        self.min_indices_integrated = np.array([np.searchsorted(
            events[i], end_time - cutoff) for i in range(self.dim)])

    def loss(self, theta):
        baseline = theta[:self.dim]
        adj = np.reshape(theta[self.dim:], (self.dim, self.dim))
        # Compute the loglikelihood
        loglik = _loglikelihood(
            baseline, adj, self.decay, self.decay_neg,
            self.gamma, self.dim, self.events, self.end_time,
            self.min_indices, self.max_indices, self.min_indices_integrated
        )
        return -(loglik + self.dim * self.end_time) / self.num_events

    def grad(self, theta):
        baseline = theta[:self.dim]
        adj = np.reshape(theta[self.dim:], (self.dim, self.dim))
        grad = _numba_grad(
            baseline, adj, self.decay, self.decay_neg,
            self.gamma, self.dim, self.events, self.end_time,
            self.min_indices, self.max_indices, self.min_indices_integrated
        )
        return grad
