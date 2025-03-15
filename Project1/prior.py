import numpy as np
from scipy.special import erfinv

U_MEAN = 1.16389876649
ERFINV_95 = erfinv(0.95)
U_STD = (0.1 * U_MEAN / (2 * np.sqrt(2) * ERFINV_95))
C_STD = (0.01 * U_MEAN / (2 * np.sqrt(2) * ERFINV_95))


def prior_U(q):
    """
    Returns the log of the prior probability distribution P(q|X) evaluated for the given value of q.
    """
    return -0.5 * (np.log(2 * np.pi * U_STD**2) + (q - U_MEAN)**2 / (U_STD**2))


def prior_C(C):
    """
    Returns the log of the prior probability distribution P(C|X) evaluated for the given value of C.
    """
    return -0.5 * (np.log(2 * np.pi * C_STD**2) + C**2 / (C_STD**2))


def prior_p(p):
    """
    Returns the log of the prior probability distribution P(p|X) evaluated for the given value of p.
    """
    if p < 1 or p > 10:
        return -np.inf
    return -np.log(9)


def prior(q, C, p):
    """
    Returns the log of the prior probability distribution P(q,C,p|X) evaluated for the given values of q, C, p.
    """
    # Validate input values
    if p < 0:
        return -np.inf
    return prior_U(q) + prior_C(C) + prior_p(p)
