import numpy as np
import scipy.stats as stats


def log_prior_normal(x, mean, std):
    return stats.norm.logpdf(x, loc=mean, scale=std)


def log_prior_uniform(x, low, high):
    if low <= x <= high:
        return 0
    else:
        return -np.inf


def prior_alpha(alpha):
    return stats.beta.logpdf(alpha, a=2, b=2)
    # return log_prior_uniform(alpha, 0, 1)


def prior_f0(f0):
    return log_prior_uniform(f0, 0, 1)


def prior_f1(f1):
    return log_prior_uniform(f1, -0.005, 0.005)


def prior(alpha, f0, f1):
    """
    Returns the log of the prior probability distribution evaluated at alpha, f0, f1.
    """
    # Validate input values
    return prior_alpha(alpha) + prior_f0(f0) + prior_f1(f1)
