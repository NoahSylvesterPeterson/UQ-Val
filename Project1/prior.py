
#!/usr/bin/env python
#
import numpy as np
from scipy.stats import norm, uniform
from scipy.special import erfinv

U_MEAN = 1.16389876649
ERFINV_95 = erfinv(0.95)
U_STD = (0.1 * U_MEAN / (2 * np.sqrt(2) * ERFINV_95))
C_STD = (0.01 * U_MEAN / (2 * np.sqrt(2) * ERFINV_95))
LOG_SQRT_2PI = np.log(np.sqrt(2 * np.pi))


def prior_U(q):
    """
    This routine should return the log of the
    prior probability distribution: P(q|X)
    evaluated for the given value of q.
    """

    return norm.logpdf(q, loc=U_MEAN, scale=U_STD)


def prior_C(C):
    """
    This routine should return the log of the
    prior probability distribution: P(C|X)
    evaluated for the given value of C.
    """
    return norm.logpdf(C, loc=0, scale=C_STD)


def prior_p(p):
    """
    This routine should return the log of the
    prior probability distribution: P(p|X)
    evaluated for the given value of p.
    """
    return uniform.logpdf(p, loc=1, scale=9)
#
# One should not have to edit the routine below
#


def prior(q, C, p):
    """
    This routine should return the log of the
    prior probability distribution: P(q,C,p|X)
    evaluated for the given values of q, C, p.
    """
    # for some reason the p guesses are sometimes negative, this
    # patches that up
    if (p < 0):
        return -np.inf
    return prior_U(q) + prior_C(C) + prior_p(p)
