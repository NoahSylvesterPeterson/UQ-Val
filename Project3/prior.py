import numpy as np
import scipy.stats as stats


def log_prior_normal(x, mean, var):
    return -0.5 * (np.log(var) + (x - mean) ** 2 / var)


def log_prior_uniform(x, low, high):
    if low <= x <= high:
        return 0
    else:
        return -np.inf


def prior_alpha(alpha):
    return log_prior_normal(alpha, mean=2.94230780e-01, var=2.94230780e-01**2)
    # return stats.beta.logpdf(alpha, a=2, b=2)
    # return log_prior_uniform(alpha, 0, 1)


def prior_f0(f0):
    return log_prior_normal(f0, mean=6.87644595e-01, var=6.87644595e-01**2)
    # return log_prior_uniform(f0, 0, 1)


def prior_f1(f1):
    return log_prior_normal(f1, mean=1.69095166e-04, var=1.69095166e-04**2)
    # return log_prior_uniform(f1, -0.005, 0.005)


def prior_f2(f1):
    return log_prior_uniform(f1, -2.5e-5, 2.5e-5)


def prior(model, params):
    bounds = [
        (0, 1),  # alpha
        (0, 1),  # f0
        (-0.005, 0.005),  # f1
        (-2.5e-5, 2.5e-5)  # f2
    ]
    return sum(stats.truncnorm.logpdf(p, loc=mu, scale=np.sqrt(np.abs(mu)), a=(
        b[0] - mu) / np.sqrt(np.abs(mu)), b=(b[1] - mu) / np.sqrt(np.abs(mu))) for p, mu, b in zip(params, model.mle_means, bounds))
