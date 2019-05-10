import numpy as np

from tick.prox import ProxL2Sq, ProxL1

from .monitor import Monitor
from ..model import ModelHawkesExpKernCondLogLikSyncNoise
from .. import util


class HawkesExpKernConditionalMLE(object):

    _available_solvers = {
        "sgd",
    }

    def __init__(self, decay, n_threads=1, approx_type='smooth',
                 decay_neg=100.0, gamma=1000.0, epsilon=1e-3,
                 hawkes_penalty='l1', hawkes_base_C=1e3, hawkes_adj_C=1e3,
                 noise_penalty='l2', noise_C=1e4,
                 solver='sgd', n_chunks=10,
                 max_iter=1000, tol=1e-5,
                 step_z=1e-2, step_theta=1e-2,
                 verbose=True, print_every=100, record_every=100):
        self.decay = decay

        if solver not in self._available_solvers:
            raise ValueError("``solver`` must be one of [%s], got %s" %
                             (', '.join(self._available_solvers), solver))
        self.solver = solver

        self.n_chunks = n_chunks
        self.max_iter = max_iter
        self.tol = tol
        self._n_iter_done = 0

        if callable(step_z):
            self.step_z = step_z
        else:
            self.step_z = lambda t: step_z

        if callable(step_theta):
            self.step_theta = step_theta
        else:
            self.step_theta = lambda t: step_theta

        assert noise_penalty == 'l2'
        assert hawkes_penalty == 'l1'

        self.noise_C = noise_C
        self.hawkes_base_C = hawkes_base_C
        self.hawkes_adj_C = hawkes_adj_C

        self.prox_noise = ProxL2Sq(strength=1/self.noise_C)
        self.prox_base = ProxL2Sq(strength=1/self.hawkes_base_C)
        self.prox_adj = ProxL1(strength=1/self.hawkes_adj_C)

        self.model_obj = ModelHawkesExpKernCondLogLikSyncNoise(
            decay=decay, n_threads=n_threads, approx_type=approx_type,
            decay_neg=decay_neg, gamma=gamma, epsilon=epsilon
        )

        self._fitted = False

        self._verbose = verbose
        self._print_every = print_every
        self._record_every = record_every
        self._init_monitor()

    def _split_data_into_chunks(self, events, end_time):
        """
        Split the data into chunks for SGD
        """
        events_multi = list()
        x = np.linspace(0, end_time, self.n_chunks + 1)
        for start_time_i, end_time_i in zip(x[:-1], x[1:]):
            events_multi.append(list(map(lambda e: e[
                                                (e >= start_time_i) &
                                                (e < end_time_i)
                                            ] - start_time_i, events)))
        end_time_multi = np.diff(x)[0] + 1e-5
        return events_multi, end_time_multi

    def _set_data(self, events, end_time):
        """
        Set the observed data variables
        """
        if not (isinstance(events, list) and
                isinstance(events[0], np.ndarray) and
                (len(events[0].shape) == 1)):
            raise ValueError("`events` must be a list of list of ndarray")
        if (not isinstance(end_time, (int, float))) or (end_time <= 0):
            raise ValueError("`end_time` must be a single positive number")
        self.events, self.end_time = self._split_data_into_chunks(events, end_time)
        # Number of realizations
        self.n_real = len(events)
        # Number of dimension
        self.dim = len(self.events[0])
        self.n_coeffs = self.dim * (self.dim + 2)

    def _init_monitor(self):
        self._monitor = Monitor(self._verbose, self._print_every,
                                self._record_every)
        self._monitor.set_print_order(['n_iter', 'rel_theta', 'rel_z',
                                       'z', 'theta'])
        self._monitor.set_print_style({'rel_z': '%.2e', 'rel_theta':  '%.2e',
                                       'z': '%12s', 'theta': '%s'})

    def _update_monitor(self, t, index, coeffs_t, step_z_t, step_theta_t,
                        rel_theta, rel_z):
        # Check if need to force printing the monitor
        force_print = self.has_converged or (t == self.max_iter)
        should_update = force_print or (t % self._record_every == 0)
        if should_update:
            self._monitor.receive(n_iter=t, index=index,
                                  rel_theta=float(rel_theta),
                                  rel_z=float(rel_z),
                                  z=coeffs_t[:self.dim].round(2).tolist(),
                                  theta=coeffs_t[self.dim:].round(2).tolist(),
                                  has_converged=int(self.has_converged),
                                  force_print=force_print)

    def _has_converged(self, t, coeffs_old, coeffs_new):
        # Compute the distance of Hawkes parameters between updates
        rel_theta = util.relative_distance(
            coeffs_old[self.dim:], coeffs_new[self.dim:], norm=1
        )
        # Compute the distance of Noise variables between updates
        rel_z = util.relative_distance(
            coeffs_old[:self.dim], coeffs_new[:self.dim], norm=1
        )
        # Stop if the `rel_theta` have converged
        self.has_converged = rel_theta < self.tol

        return self.has_converged, rel_theta, rel_z

    def _sgd_solver(self, z_start, theta_start, seed=None, callback=None):
        if seed is not None:
            np.random.seed(seed)
        coeffs_old = np.hstack((z_start, theta_start))
        for t in range(self.max_iter + 1):
            # Sample a realization
            i = np.random.randint(0, self.n_real)
            # Fit the model to these events
            self.model_obj.fit(self.events[i], self.end_time)
            # Compute the gradient
            grad_t = self.model_obj.grad(coeffs_old)

            # Keep track of previous coefficients
            z_old = coeffs_old[:self.dim]
            theta_old = coeffs_old[self.dim:]

            # Step size
            step_z_t = self.step_z(t)
            step_theta_t = self.step_theta(t)

            # Do a gradient step
            coeffs_new = np.zeros_like(coeffs_old)
            coeffs_new[:self.dim] = self.prox_noise.call(
                    coeffs_old[:self.dim] - step_z_t * grad_t[:self.dim])
            coeffs_new[self.dim:2*self.dim] = self.prox_base.call(
                coeffs_old[self.dim:2*self.dim] - step_z_t * grad_t[self.dim:2*self.dim])
            coeffs_new[2*self.dim:] = self.prox_adj.call(
                coeffs_old[2*self.dim:] - step_z_t * grad_t[2*self.dim:])

            # Project Hawkes parameters onto the positive plane
            coeffs_new[self.dim:] = np.clip(
                coeffs_new[self.dim:], a_min=1e-3, a_max=100.0)

            # Check convergence
            has_converged, rel_theta, rel_z = self._has_converged(
                t, coeffs_old, coeffs_new)

            # Update the algorithm history monitor
            self._update_monitor(t, i, coeffs_new, step_z_t, step_theta_t,
                                 rel_theta, rel_z)

            self._n_iter_done = t
            self.coeffs = coeffs_new

            coeffs_old = coeffs_new.copy()
            if has_converged:
                break

            if callback is not None:
                callback(self)

        return coeffs_new

    def fit(self, events, end_time, z_start, theta_start, seed=None,
            callback=None):
        self._fitted = True
        # Set the observed data
        self._set_data(events, end_time)
        # Run the solver
        if self.solver == 'sgd':
            self.coeffs = self._sgd_solver(
                z_start, theta_start, seed, callback)
        else:
            raise ValueError('Invalid solver')
        # Format nicely the output parameters
        self.noise = self.coeffs[:self.dim]
        self.baseline = self.coeffs[self.dim:2*self.dim]
        self.adjacency = np.reshape(
            self.coeffs[2*self.dim:], (self.dim, self.dim)
        )
