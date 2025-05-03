#!/bin/py
from matplotlib.ticker import FormatStrFormatter
from multiprocessing import Pool
from math import ceil, floor
from emcee import EnsembleSampler, autocorr, moves
import emcee
import likelihood
import prior
import importance_sampling
from scipy import stats, optimize
from matplotlib import pyplot as plt
import numpy as np
from cmcrameri import cm
import os
import corner
import pandas as pd
from rich import print
from rich.traceback import install
install(show_locals=True)

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams[
    "text.latex.preamble"
] = r"""
\usepackage[T1]{fontenc}
\usepackage{newpxtext,newpxmath}
\usepackage{amsmath}
\usepackage{bm}
"""

default_cycler = plt.cycler('color', cm.romaO(np.linspace(0, 1, 8)))
plt.rcParams['axes.prop_cycle'] = default_cycler
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams["font.size"] = 12

# local files that will be imported

# construct map of prior functions, to plot below
priors = [prior.prior_alpha, prior.prior_f0, prior.prior_f1, prior.prior_f2]
prior_names = ['alpha', 'f0', 'f1', 'f2']
fdict = dict(zip(prior_names, priors))


def plotter(chain, quant, xmin=None, xmax=None):
    """subroutine that generates a .pdf file plotting a quantity"""
    bins = np.linspace(np.min(chain), np.max(chain), 200)
    qkde = stats.gaussian_kde(chain)
    qpdf = qkde.evaluate(bins)

    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')

    # plot prior (requires some cleverness to do in general)
    qpr = [fdict[quant](x) for x in bins]
    qpri = [np.exp(x) for x in qpr]
    qpri = qpri
    ax2 = plt.twinx(plt.gca())
    ax2.plot(bins, qpri, 'k', linewidth=3, label=f"Prior $\\pi_0({quant})$", zorder=1)
    ax2.set_ylabel(f'$\\pi_0({quant})$', fontsize=16)

    # plot posterior
    ax.plot(bins, qpdf, 'mediumpurple', linewidth=3, label=f"Post $\\pi({quant})$", zorder=10)
    ax.set_ylabel(f'$\\pi({quant})$', fontsize=16)

    ax.set_zorder(2)
    ax2.set_zorder(1)
    ax.patch.set_visible(False)

    # user specified bounds to x-range:
    if (xmin is not None and xmax is not None):
        bounds = np.array([xmin, xmax])
        plt.xlim(bounds)

    ax2.set_xlabel(f'${quant}$', fontsize=16)
    ax.set_xlabel(f'${quant}$', fontsize=16)
    fig.legend(loc='outside upper center', fontsize=16, ncol=2)
    plt.savefig(quant + '_post.pdf', bbox_inches='tight')
    plt.close()


def sample_walkers(model, Y_CO2, nsamples, flattened_chain):
    Y_CO2 = np.array([Y_CO2]) if np.isscalar(Y_CO2) else np.asarray(Y_CO2)
    models = np.zeros((nsamples, len(Y_CO2)))
    draw = np.floor(np.random.uniform(0, len(flattened_chain), size=nsamples)).astype(int)
    params = flattened_chain[draw]
    for i, param_set in enumerate(params):
        models[i, :] = model.predict(param_set, Y_CO2)
    spread = np.std(models, axis=0)
    med_model = np.median(models, axis=0)
    return med_model, spread, models


class BayesianRichardsonExtrapolation():
    """Computes the Bayesian Richardson extrapolation posterior log density."""

    def __init__(self, model: likelihood.Model):
        self.model = model

    def __call__(self, params, dtype=np.double):
        log_prior = prior.prior(self.model, params)
        log_likelihood = self.model.log_likelihood(params)
        if not np.isfinite(log_prior) or not np.isfinite(log_likelihood):
            return -np.inf
        return log_likelihood + log_prior


def textual_boxplot(label, unordered, header):
    n, d = np.size(unordered), np.sort(unordered)
    if (header):
        print((10 * " %15s") % ("", "min", "P5", "P25", "P50", "P75", "P95", "max", "mean", "stddev"))
    print((" %15s" + 9 * " %+.8e") % (label,
                                      d[0],
                                      d[[floor(1. * n / 20), ceil(1. * n / 20)]].mean(),
                                      d[[floor(1. * n / 4), ceil(1. * n / 4)]].mean(),
                                      d[[floor(2. * n / 4), ceil(2. * n / 4)]].mean(),
                                      d[[floor(3. * n / 4), ceil(3. * n / 4)]].mean(),
                                      d[[floor(19. * n / 20), ceil(19. * n / 20)]].mean(),
                                      d[-1],
                                      d.mean(),
                                      d.std()))
    # return d[[floor(1.*n/20), ceil(1.*n/20)]].mean(), d[[floor(17.*n/20), ceil(17.*n/20)]].mean()
    return d.mean(), 2 * d.std()


def main(model, lsqr_params):
    # initialize the Bayesian Calibration Procedure
    print("\nInitializing walkers")
    filename = "mcmc.h5"
    nwalk = 6 * 24
    ndim = model.num_params

    # initial guesses for the walkers starting locations
    map_estimate = likelihood.compute_mle(model, likelihood.initial_guesses[:ndim]).x

    params0 = np.tile(map_estimate, nwalk).reshape(nwalk, ndim)
    params0[:, 0] += np.random.randn(nwalk) * 0.1 * map_estimate[0]    # Perturb alpha
    params0[:, 0] = np.clip(params0[:, 0], 0, 1)
    params0[:, 1] += np.random.randn(nwalk) * 0.1 * map_estimate[1]   # Perturb f0
    params0[:, 1] = np.clip(params0[:, 1], 0, 1)
    if ndim > 2:
        params0[:, 2] += np.random.randn(nwalk) * 0.1 * map_estimate[2]   # Perturb f1
        params0[:, 2] = np.clip(params0[:, 2], -0.005, 0.005)
    if ndim > 3:
        params0[:, 3] += np.random.randn(nwalk) * 0.1 * map_estimate[3]   # Perturb f2
        params0[:, 3] = np.clip(params0[:, 3], -2.5e-5, 2.5e-5)

    print("\nInitializing the sampler and burning in walkers")

    backend = emcee.backends.HDFBackend(filename, name=f"{model}")
    backend.reset(nwalk, ndim)

    with Pool(24) as pool:
        sampler = BayesianRichardsonExtrapolation(model=model)
        s = EnsembleSampler(nwalk, ndim, sampler, pool=pool, backend=backend, moves=[
            moves.DEMove(),
            moves.DESnookerMove()
        ])
        pos, _, _ = s.run_mcmc(params0, 4000, progress=True, tune=True)
        s.reset()
        finished = False
        step = 0
        step_size = 2000
        print("\nSampling the posterior density for the problem")
        while not finished:
            # Use autocorrelation time to determine if we have converged
            # adapted from https://groups.google.com/g/emcee-users/c/KuoXbQTH_8Q
            pos = s.run_mcmc(pos, step_size, progress=True, tune=True)
            step += step_size
            try:
                tau = s.get_autocorr_time()

                if len(tau) != ndim:
                    s.reset()
                    print(f"not enough burn-in steps, resetting")
                    step_size += 5000
                    continue
                if np.any(np.isnan(tau)):
                    print(f"step {step}  progress {step/(tau*50)} (acceptance fraction {s.acceptance_fraction.mean()})")
                    s.reset()
                    step_size += 5000
                    print(f"not enough burn-in steps, resetting")
                    continue

                print(f"step {step}  progress {step/(tau*50)} (acceptance fraction {s.acceptance_fraction.mean()})")
                if not np.any(step / (tau * 50) < 1):
                    finished = True
            except autocorr.AutocorrError as err:
                tau = err.tau
                if len(tau) != ndim:
                    step = 0
                    s.reset()
                print(f"{tau=}")
                print(f"step {step}  progress {step/(tau*50)}")

        # final sampling
        s.run_mcmc(pos, 4000, progress=True, tune=True)
        print(f'{tau=}')

        print(f"Mean acceptance fraction was {s.acceptance_fraction.mean():.3f}")

        # 1d Marginals
        print("\nDetails for posterior one-dimensional marginals:")

        # Remove a sufficient number of burn-in steps
        burn = int(np.ceil(np.max(tau)) * 2)
        flat_samples = s.get_chain(discard=burn, flat=True)

        for i in range(ndim):
            textual_boxplot(prior_names[i], flat_samples[:, i], header=True)

        # FIGURES: Marginal posterior(s)
        print("\nPrinting PDF output")
        for i in range(ndim):
            plotter(flat_samples[:, i], prior_names[i])

        means = np.mean(flat_samples, axis=0)
        print(f"\nPosterior means: {means}")
        # FIGURE: Joint posterior(s)
        full_labels = ["$\\alpha$", "$f_0$", "$f_1$", "$f_2$"]
        figure = corner.corner(
            flat_samples,
            labels=full_labels[:ndim],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".3g",
            title_kwargs={"fontsize": 12}
        )
        axes = np.array(figure.axes).reshape((ndim, ndim))
        # Loop over the diagonal
        for i in range(ndim):
            ax = axes[i, i]
            ax.axvline(means[i], color="r")

        # Loop over the histograms
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(means[xi], color="r")
                ax.axhline(means[yi], color="r")
                ax.plot(means[xi], means[yi], "sr")

        plt.savefig('joint_post_corner.pdf', bbox_inches='tight')
        plt.close()

        # Make predictions
        best_i = np.argmax(s.flatlnprobability)
        max_likelihood_params = flat_samples[best_i]
        best_fit_model = model.evaluate(max_likelihood_params)
        print(f"\nBest fit model: " + ", ".join([f'{name}={val}' for name,
              val in zip(prior_names, max_likelihood_params)]))
        print(f"\tLog Probability: {s.flatlnprobability[best_i]}")

        fig, ax = plt.subplots(figsize=(6, 4))
        median, spread, models = sample_walkers(model, likelihood.CO2, 400, flat_samples)
        mean = np.mean(models, axis=0)
        print(spread)
        plt.errorbar(likelihood.CO2, likelihood.TEMPERATURE, yerr=2 * likelihood.STDDEV_T, fmt='o', color='black',
                     markersize=5, label='Observed data with 2$\\sigma$ error bars', zorder=1)
        plt.fill_between(
            likelihood.CO2,
            median - 2 * spread,
            median + 2 * spread,
            color='grey',
            alpha=0.5,
            label=r'$2\sigma$ Posterior Spread', zorder=2)
        plt.plot(likelihood.CO2, mean, color='orange', label='Posterior predictive mean', zorder=10)
        plt.plot(
            likelihood.CO2,
            model.evaluate(lsqr_params),
            color='blue',
            linestyle='--',
            label='Least squares model', zorder=9)
        plt.xlabel('CO2 concentration (ppm)')
        plt.ylabel('Temperature (K)')
        plt.legend()
        plt.savefig('posterior.pdf', bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(6, 4))
        plt.fill_between(
            likelihood.CO2,
            median - 2 * spread,
            median + 2 * spread,
            color='grey',
            alpha=0.5,
            label=r'$2\sigma$ Posterior Spread')
        plt.errorbar(likelihood.CO2, likelihood.TEMPERATURE, yerr=2 * likelihood.STDDEV_T, fmt='o', color='black',
                     markersize=5, label='Observed data with 2$\\sigma$ error bars', zorder=1)
        plt.plot(
            likelihood.CO2,
            model.evaluate(lsqr_params),
            color='blue',
            linestyle='--',
            label='Least squares model', zorder=9)
        plt.plot(likelihood.CO2, best_fit_model, color='orange', label='Highest likelihood model', zorder=10)
        plt.xlabel('CO2 concentration (ppm)')
        plt.ylabel('Temperature (K)')
        # plt.title('Posterior predictive distribution', fontsize=16)
        plt.legend()
        plt.savefig('posterior_best_fit.pdf', bbox_inches='tight')
        plt.close()

        pred_CO2 = [717.0]
        pred_T, stdev_T, models_T = sample_walkers(model, pred_CO2, 400, flat_samples)
        pred_T_mean = np.mean(models_T, axis=0)
        print(f"Predicted temperature for CO2 concentration {pred_CO2} ppm:\n")
        print(f"\tMedian: {pred_T[0]:.6g} ± {2*stdev_T[0]:.6g} K")
        print(f"\tMean:   {pred_T_mean[0]:.6g} ± {2*stdev_T[0]:.6g} K")
        return means


def vis_data():
    data = likelihood.read_data()
    plt.plot(data)
    plt.savefig('./tmp.png')


def problem1a(alpha, f):
    print(f"[bold]Problem 1a: For {alpha=}, {f=}, predicted T={likelihood.MODEL_T_CONSTANT.evaluate((alpha, f))}")


def problem1b():
    print("\n[bold]Problem 1b: Least squares optimization")
    model = likelihood.MODEL_T_LINEAR
    init = (0, 0, 0)
    res: optimize.OptimizeResult = optimize.least_squares(
        lambda x: likelihood.TEMPERATURE - model.evaluate(x),
        init,
        bounds=([0, 0, -np.inf], [1, 1, np.inf]),
    )
    if res.success:
        print("\tOptimization was successful.")
    else:
        print("\tOptimization failed.")
    alpha, f0, f1 = res.x
    print(f"\tParameters: {alpha=}, {f0=}, {f1=}")
    print(f"\tCost function value:", res.cost)
    print(f"\tMRSQ:", np.linalg.norm(res.fun) / len(likelihood.TEMPERATURE))
    print(f"\t-Log likelihood:", -model.log_likelihood(res.x))
    f = model.embedded_f(res.x, likelihood.CO2)
    if np.any(f < 0) or np.any(f > 1):
        print(f"\t[yellow]Warning: {len(f[f > 1]) + len(f[f<0])} f values are out of bounds!")
    return alpha, f0, f1


def plot_analytical_calibration(model, nsample=100000):
    # Plot the analytical calibration
    posterior: stats.rv_continuous = model.linear_model(model.mle_means)
    temp_chain = posterior.rvs(size=nsample)
    print(temp_chain.shape)
    means = posterior.mean
    print(f"\nPosterior means: {means}")
    # FIGURE: Joint posterior(s)
    full_labels = ["$\\alpha$", "$f_0$", "$f_1$", "$f_2$"]
    figure = corner.corner(
        temp_chain,
        labels=full_labels[:model.num_params],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3g",
        title_kwargs={"fontsize": 12}
    )
    axes = np.array(figure.axes).reshape((model.num_params, model.num_params))
    # Loop over the diagonal
    for i in range(model.num_params):
        ax = axes[i, i]
        ax.axvline(means[i], color="r")

    # Loop over the histograms
    for yi in range(model.num_params):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(means[xi], color="r")
            ax.axhline(means[yi], color="r")
            ax.plot(means[xi], means[yi], "sr")

    plt.savefig('analytical_joint_post_corner.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    problem1a(0.3, 0.6)
    params = problem1b()
    pd.set_option('display.max_rows', None)
    # print(read_data().head(81))
    # vis_data()
    # for model in [likelihood.MODEL_T_CONSTANT, likelihood.MODEL_T_QUADRATIC, likelihood.MODEL_T_LINEAR]:
    for model in [likelihood.MODEL_T_CONSTANT, likelihood.MODEL_T_QUADRATIC, likelihood.MODEL_T_LINEAR]:
        analytical = model.linear_model(model.mle_means)
        sample_T = np.empty((10000,))
        i = 0
        while i < 10000:
            sample = analytical.rvs(1)
            if sample[0] < 0 or sample[0] > 1 or sample[1] < 0 or sample[1] > 1:
                continue
            f = model.embedded_f(sample, likelihood.CO2)
            if np.any(f < 0) or np.any(f > 1):
                continue
            sample_T[i] = model.predict(sample, np.array([717.0]))[0]
            i += 1
        pred_T = np.mean(sample_T)
        std_T = np.std(sample_T)
        print(f"Prediction (linearized) for {model}: {model.predict(analytical.mean, [717.0])}")
        print(f"Prediction from samples (linearized) for {model}: {pred_T} ± {2 * std_T} K")
    # mcmc_means = main(likelihood.MODEL_T_LINEAR, analytical.mean)
    # print(f"MCMC MAP for {model}: {mcmc_means}")
    plot_analytical_calibration(model)
    importance_sampling.compare_models()
    # print("MCMC final -log likelihood function: ", -model.log_likelihood(mean_params))
