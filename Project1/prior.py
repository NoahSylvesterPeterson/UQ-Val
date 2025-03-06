
#!/usr/bin/env python
#
import numpy as np


def prior_U(q):
    """
    This routine should return the log of the
    prior probability distribution: P(q|X)
    evaluated for the given value of q.
    """
    return prior_U


def prior_C(C):
    """
    This routine should return the log of the
    prior probability distribution: P(C|X)
    evaluated for the given value of C.
    """
    return prior_C


def prior_p(p):
    """
    This routine should return the log of the
    prior probability distribution: P(p|X)
    evaluated for the given value of p.
    """
    return prior_p
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
        return -1 * np.inf
    return prior_U(q) + prior_C(C) + prior_p(p)
