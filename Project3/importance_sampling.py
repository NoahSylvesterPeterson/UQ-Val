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
    means = likelihood.prior_means
    variances = [m * 0.001 for m in means]  # Example variances for the parameters
    prior_dists = [stats.Normal(mu=mu, sigma=np.sqrt(var)) for mu, var in zip(means, variances)]
    samples = np.array([p.sample(10000) for p in prior_dists]).T
    likelihood_dist = [np.exp(model.linear_log_likelihood(s)) for s in samples]
    prior_probs = np.asarray([[p.pdf(s) for p, s in zip(prior_dists, sample)] for sample in samples])
    posterior_probs = np.asarray([l * np.asarray([p.pdf(s) for p, s in zip(prior_dists, sample)])
                                 for l, sample in zip(likelihood_dist, samples)])
    posterior_probs /= np.sum(posterior_probs, axis=0)
    posterior_means = np.mean(posterior_probs, axis=0)
    posterior_vars = np.var(posterior_probs, axis=0)

    # print posteriors
    for i, (mean, var) in enumerate(zip(posterior_means, posterior_vars)):
        print(f"Posterior {i}: Mean: {mean}, Variance: {var}")

    print(f"Mean: {mean}, Variance: {var}")
    return stats.norm(loc=mean, scale=np.sqrt(var))


def compare_models():
    pass
