import numpy as np
from scipy.constants import Stefan_Boltzmann

S = 1370  # W/m^2
R = 6.38e6  # m


def embedded_f_linear(f0, f1, Y_C02):
    return f0 + f1 * Y_C02


def model_T(alpha, f0, f1, Y_C02):
    """
    Returns the temperature T for a given alpha, f0, and f1.
    """
    f = embedded_f_linear(f0, f1, Y_C02)
    return np.power((S / 4) * (1 - alpha) / (Stefan_Boltzmann * (1 - 0.5 * f)), 0.25)


def likelihood(alpha, f0, f1):
    """
    Returns the log of the likelihood function alpha, f0, f1
    """
    pass
