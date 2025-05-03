import emcee
from scipy import stats
import numpy as np
import likelihood
import prior
from functools import reduce


def importance_sampling(model, num_samples=1000):
    """
    Guess the importance distribution based on the analytical calibration.

    Returns:
    scipy.stats.rv_frozen: The frozen random variable representing the importance distribution.
    """
    analytical_post = model.linear_model(model.mle_means)
    weights = np.zeros(num_samples)
    i = 0
    while i < num_samples:
        sample = analytical_post.rvs(1)
        if sample[0] < 0 or sample[0] > 1:
            continue
        f = model.embedded_f(sample, likelihood.CO2)
        if np.any(f < 0) or np.any(f > 1):
            continue
        log_prior = prior.prior(model, sample)
        log_likelihood = model.log_likelihood(sample) + 2000
        log_posterior = log_prior + log_likelihood
        log_q = analytical_post.logpdf(sample)
        if not np.isfinite(log_q):
            continue
        log_posterior -= log_q
        weights[i] = np.exp(log_posterior)
        i += 1
    evidence = np.sum(weights) / num_samples
    return evidence


def compare_models():
    for model in [likelihood.ModelConstant(), likelihood.ModelLinear(), likelihood.ModelQuadratic()]:
        # Calculate the importance sampling evidence
        evidence = importance_sampling(model, num_samples=1000)
        print(f"Model: {model}, Evidence: {evidence}")
