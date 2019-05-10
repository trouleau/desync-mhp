import numpy as np


def compute_mean_intensity(adjacency, baseline, decays):
    """
    Compute the mean intensity of a Hawkes process with Exponential excitation
    function defined according to the `tick` library definition
    """
    n_nodes = len(baseline)
    # Do not divide by beta to match tick def. of kernels with ours
    extr_func_l1_norm = adjacency
    intr_func_l1_norm = baseline
    meanint = np.linalg.inv(
        np.eye(n_nodes) - extr_func_l1_norm
    ).dot(intr_func_l1_norm)
    return meanint


def compute_kernels_integral(amplitudes, basis_kernels, kernel_dt):
    return np.sum(np.dot(amplitudes, basis_kernels) * kernel_dt, axis=-1)


def compute_num_err(x_est, x_true, thresh):
    return int(np.sum((x_est > thresh) ^ (x_true > 0.0)))


def relative_distance(x_new, x_old, norm=1):
    x_old_norm = np.linalg.norm(x_old, ord=norm)
    x_diff_norm = np.linalg.norm(x_new - x_old, ord=norm)
    return x_diff_norm / x_old_norm


def build_theta(baseline, adjacency):
    return np.hstack((baseline, adjacency.ravel()))
