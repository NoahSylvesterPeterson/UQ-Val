import emcee
from scipy import stats
import numpy as np
import likelihood
import prior
from functools import reduce


def load_sampled_chain():
    """
    Load a sampled chain from an HDF5 file using emcee's backend.

    Returns:
    emcee.backends.HDFBackend: The backend object containing the sampled chain.
    """
    # Load the sampled chain from the HDF5 file
    return emcee.backends.HDFBackend("mcmc.h5")


def guess_importance_distribution(model):
    """
    Guess the importance distribution based on the analytical calibration.

    Returns:
    scipy.stats.rv_frozen: The frozen random variable representing the importance distribution.
    """
    analytical_post = model.linear_model()
    chain = load_sampled_chain()


def compare_models():
    pass
