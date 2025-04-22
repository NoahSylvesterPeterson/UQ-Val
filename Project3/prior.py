import numpy as np
from scipy.special import erfinv


def log_prior_normal(x, mean, std):
    return -0.5 * (np.log(2 * np.pi * std**2) + (x - mean)**2 / (std**2))


def log_prior_uniform(x, low, high):
    if low <= x <= high:
        return -np.log(high - low)
    else:
        return -np.inf


def prior_alpha(alpha):
    return log_prior_uniform(alpha, 0, 1)


def prior_f0(alpha):
    return log_prior_uniform(alpha, 0, 1)


def prior_f1(alpha):
    return log_prior_uniform(alpha, 0, 1)


def prior(alpha, f0, f1):
    """
    Returns the log of the prior probability distribution evaluated at alpha, f0, f1.
    """
    # Validate input values
    return prior_alpha(alpha) + prior_f0(f0) + prior_f1(f1)
