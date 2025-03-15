#!/usr/bin/env python
#
import numpy as np

data = [
    [2.0, 1.16828362427, 0.0001283],
    [np.sqrt(2), 1.16429173392, 0.0003982],
    [1.0, 1.16367827195, 0.0001282],
    [0.5, 1.16389876649, 0.0002336],
]


def likelihood(q, C, p):
    """
    This routine should return the log of the
    likelihood function: P(qi|q,C,p,X)
    evaluated for given values of q, C and p
    """

    likelihood_sum = 0

    for data_point in data:
        h, q_i, sigma_i = data_point
        # Calculate the likelihood
        likelihood_sum += -0.5 * (np.log(2 * np.pi * sigma_i**2) + (q_i - (q - C * np.power(h, p)))**2 / sigma_i**2)

    return likelihood_sum
