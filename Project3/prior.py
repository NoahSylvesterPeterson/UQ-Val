import numpy as np
import scipy.stats as stats
from likelihood import MODEL_T_CONSTANT, MODEL_T_LINEAR, MODEL_T_QUADRATIC
from rich.traceback import install
install(show_locals=True)


def log_prior_normal(x, mean, var):
    return -0.5 * (np.log(var) + (x - mean) ** 2 / var)


def log_prior_uniform(x, low, high):
    if low <= x <= high:
        return 0
    else:
        return -np.inf


bounds = [
    (0, 1),  # alpha
    (0, 1),  # f0
    (-0.003, 0.003),  # f1
    (-9e-6, 9e-6)  # f2
]


def prior_k_trunc(model, k):
    mu_normal = model.mle_means[k]
    sigma_normal = np.sqrt(np.abs(model.mle_means[k]))
    alpha = (bounds[k][0] - mu_normal) / sigma_normal
    beta = (bounds[k][1] - mu_normal) / sigma_normal
    bias = sigma_normal * (stats.norm.pdf(beta) - stats.norm.pdf(alpha)) / \
        (stats.norm.cdf(beta) - stats.norm.cdf(alpha))
    mu = mu_normal + bias
    return stats.truncnorm(a=alpha, b=beta, loc=mu, scale=sigma_normal)


def prior_k(model, k):
    rv = stats.Normal(mu=model.mle_means[k], sigma=np.sqrt(np.abs(model.mle_means[k])))
    return rv


prior_dists = {
    "ModelConstant": {
        "alpha": prior_k_trunc(MODEL_T_CONSTANT, 0),
        "f0": prior_k(MODEL_T_CONSTANT, 1),
    },
    "ModelLinear": {
        "alpha": prior_k_trunc(MODEL_T_LINEAR, 0),
        "f0": prior_k(MODEL_T_LINEAR, 1),
        "f1": prior_k(MODEL_T_LINEAR, 2),
    },
    "ModelQuadratic": {
        "alpha": prior_k_trunc(MODEL_T_QUADRATIC, 0),
        "f0": prior_k(MODEL_T_QUADRATIC, 1),
        "f1": prior_k(MODEL_T_QUADRATIC, 2),
        "f2": prior_k(MODEL_T_QUADRATIC, 3),
    },
}


def prior(model, params):
    dists = prior_dists[f'{model}']
    param_names = ['alpha', 'f0', 'f1', 'f2']
    return sum(dists[k].logpdf(params[i]) for i, k in enumerate(param_names[:len(params)]))
