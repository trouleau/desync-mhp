#cython: boundscheck=False, wraparound=False, nonecheck=False, language_level=3

from libc.math cimport exp, log
from libc.stdlib cimport rand, RAND_MAX
import numpy as np
cimport numpy as np

DTYPE = np.int
DTYPE_double = np.double
ctypedef np.int_t DTYPE_t
ctypedef np.double_t DTYPE_double_t

cdef _search_sorted(double[:] a, double[:] v):
    """
    Find indices where elements should be inserted to maintain order.
    Find the indices into a sorted array `a` such that, if the corresponding
    elements in `v` were inserted before the indices, the order of `a` would
    be preserved.
    Parameters
    ----------
    a : 1-D array_like
        Input array, sorted in ascending order.
    v : array_like
        Values to insert into `a`.
    """
    cdef unsigned nv = v.shape[0]
    cdef DTYPE_t[:] idx = np.empty(nv, dtype=DTYPE)
    cdef unsigned na = a.shape[0]
    cdef unsigned int ia = 0
    cdef unsigned int iv
    cdef double vi
    for iv in range(nv):
        vi = v[iv]
        while True:
            if ia < na:
                if vi <= a[ia]:
                    idx[iv] = ia
                    break
                else:
                    ia += 1
            else:
                idx[iv] = na
                break
    return idx


def precompute_batch_indices(list events, float approx, float decay):
    cdef int n_dim = len(events)
    cdef int m, n, k, events_m_size
    cdef DTYPE_t[:, :] indices_m,
    cdef DTYPE_t[:] indices_mn

    cdef list max_indices = []
    for m in range(n_dim):
        events_m_size = events[m].shape[0]
        indices_m = np.empty((n_dim, events_m_size), dtype=DTYPE)
        for n in range(n_dim):
            indices_mn = _search_sorted(events[n], events[m])
            for k in range(events_m_size):
                indices_m[n, k] = indices_mn[k]
        max_indices.append(indices_m)

    cdef float cutoff = -log(approx) / decay

    cdef list min_indices = []
    for m in range(n_dim):
        events_m_size = events[m].shape[0]
        indices_m = np.empty((n_dim, events_m_size), dtype=DTYPE)
        for n in range(n_dim):
            indices_mn = _search_sorted(events[n]+cutoff, events[m])
            for k in range(events_m_size):
                indices_m[n, k] = indices_mn[k]
        min_indices.append(indices_m)

    return min_indices, max_indices


def precompute_batch_min_indices(list events, float approx, float decay):
    cdef int n_dim = len(events)
    cdef int m, n, k, events_m_size
    cdef DTYPE_t[:, :] indices_m,
    cdef DTYPE_t[:] indices_mn

    cdef float cutoff = -log(approx) / decay

    cdef list min_indices = []
    for m in range(n_dim):
        events_m_size = events[m].shape[0]
        indices_m = np.empty((n_dim, events_m_size), dtype=DTYPE)
        for n in range(n_dim):
            indices_mn = _search_sorted(events[n]+cutoff, events[m])
            for k in range(events_m_size):
                indices_m[n, k] = indices_mn[k]
        min_indices.append(indices_m)

    return min_indices


def precompute_batch_max_indices(list events, float approx, float decay):
    cdef int n_dim = len(events)
    cdef int m, n, k, events_m_size
    cdef DTYPE_t[:, :] indices_m,
    cdef DTYPE_t[:] indices_mn

    cdef float cutoff = -log(approx) / decay

    cdef list max_indices = []
    for m in range(n_dim):
        events_m_size = events[m].shape[0]
        indices_m = np.empty((n_dim, events_m_size), dtype=DTYPE)
        for n in range(n_dim):
            indices_mn = _search_sorted(events[n]-cutoff, events[m])
            for k in range(events_m_size):
                indices_m[n, k] = indices_mn[k]
        max_indices.append(indices_m)

    return max_indices


def precompute_batch_min_indices_cutoff(list events, float cutoff):
    cdef int n_dim = len(events)
    cdef int m, n, k, events_m_size
    cdef DTYPE_t[:, :] indices_m,
    cdef DTYPE_t[:] indices_mn
    cdef list min_indices = []
    for m in range(n_dim):
        events_m_size = events[m].shape[0]
        indices_m = np.empty((n_dim, events_m_size), dtype=DTYPE)
        for n in range(n_dim):
            indices_mn = _search_sorted(events[n]+cutoff, events[m])
            for k in range(events_m_size):
                indices_m[n, k] = indices_mn[k]
        min_indices.append(indices_m)
    return min_indices


def precompute_batch_max_indices_cutoff(list events, float cutoff):
    cdef int n_dim = len(events)
    cdef int m, n, k, events_m_size
    cdef DTYPE_t[:, :] indices_m,
    cdef DTYPE_t[:] indices_mn
    cdef list max_indices = []
    for m in range(n_dim):
        events_m_size = events[m].shape[0]
        indices_m = np.empty((n_dim, events_m_size), dtype=DTYPE)
        for n in range(n_dim):
            indices_mn = _search_sorted(events[n]-cutoff, events[m])
            for k in range(events_m_size):
                indices_m[n, k] = indices_mn[k]
        max_indices.append(indices_m)
    return max_indices
