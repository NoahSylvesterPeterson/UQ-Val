#!/bin/py
from matplotlib.ticker import FormatStrFormatter
from multiprocessing import Pool
from math import ceil, floor
from emcee import EnsembleSampler, autocorr
import likelihood
import prior
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

os.environ["OMP_NUM_THREADS"] = "1"

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
fdict = {'prior_alpha': prior.prior_alpha, 'prior_f0': prior.prior_f0, 'prior_f1': prior.prior_f1}


def ReadSiple2():
    # pd.read_csv("siple_1", usecols=(3, 4), names=["year", "CO2"])
    data_old = pd.read_csv("data/siple_2.csv", usecols=(1, 2), header=0, names=["year", "CO2"])
    return (data_old)


def read_atmospheric_CO2_data():
    data_new = pd.read_csv("data/Maunoa_Lao_atmospheric_CO2_levels_1956-2024.csv",
                           skiprows=44, usecols=(0, 1), header=0, names=["year", "CO2"])
    data_old = ReadSiple2()
    data = pd.concat([data_old, data_new], ignore_index=True)
    return data


def read_temperature_data():
    """Reads the temperature data from the file and returns it as a numpy array."""
    data: pd.DataFrame = pd.read_csv("data/globalTemperatureAnomoly1900-2024.csv",
                                     skiprows=4, usecols=(0, 1), header=0, names=["year", "anomaly"])
    data["temperature"] = data["anomaly"] + 286.9

    return data[["year", "temperature"]]


def read_data():
    temp_data = read_temperature_data()
    co2_data = read_atmospheric_CO2_data()
    temp_data["year"] = pd.to_datetime(temp_data["year"], format="%Y")
    co2_data["year"] = pd.to_datetime(co2_data["year"], format="%Y")
    temp_data.set_index("year", inplace=True)
    co2_data.set_index("year", inplace=True)
    data = pd.merge(co2_data, temp_data, left_index=True, right_index=True)
    return data


def plotter(chain, quant, xmin=None, xmax=None):
    """subroutine that generates a .pdf file plotting a quantity"""
    bins = np.linspace(np.min(chain), np.max(chain), 200)
    qkde = stats.gaussian_kde(chain)
    qpdf = qkde.evaluate(bins)

    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')

    # plot prior (requires some cleverness to do in general)
    qpr = [fdict['prior_' + quant](x) for x in bins]
    qpri = [np.exp(x) for x in qpr]
    qpri = qpri / np.linalg.norm(qpri)
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


class BayesianRichardsonExtrapolation():
    """Computes the Bayesian Richardson extrapolation posterior log density."""

    def __call__(self, params, dtype=np.double):
        return (
            prior.prior(*params) +
            likelihood.likelihood(*params)
        )


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


def main():
    # initialize the Bayesian Calibration Procedure
    print("\nInitializing walkers")
    nwalk = 100

    # initial guesses for the walkers starting locations
    guess_alpha = 0.5
    guess_f0 = 0.5
    guess_f1 = 0.5

    params0 = np.tile([guess_alpha, guess_f0, guess_f1], nwalk).reshape(nwalk, 3)
    params0[:, 0] += np.random.randn(nwalk) * 0.5    # Perturb alpha
    params0[:, 1] += np.random.randn(nwalk) * 0.5    # Perturb f0
    params0[:, 2] += np.random.randn(nwalk) * 0.5    # Perturb f1
    params0 = np.clip(params0, 0, 1)

    print("\nInitializing the sampler and burning in walkers")
    with Pool(10) as pool:
        sampler = BayesianRichardsonExtrapolation()
        s = EnsembleSampler(nwalk, params0.shape[-1], sampler, pool=pool)
        pos, prob, state = s.run_mcmc(params0, 15000, progress=True)
        s.reset()
        finished = False
        step = 0
        step_size = 5000
        print("\nSampling the posterior density for the problem")
        while not finished:
            # Use autocorrelation time to determine if we have converged
            # adapted from https://groups.google.com/g/emcee-users/c/KuoXbQTH_8Q
            pos, _, _ = s.run_mcmc(pos, step_size, progress=True)
            step += step_size
            try:
                tau = s.get_autocorr_time()

                print(f"step {step}  progress {step/(tau*50)}")

                if not np.any(np.isnan(tau)):
                    finished = True

            except autocorr.AutocorrError as err:
                tau = err.tau

                print(f"step {step}  progress {step/(tau*50)}")

        # final sampling
        s.run_mcmc(pos, 10000, progress=True)
        print(f'{tau=}')

        print("Mean acceptance fraction was %.3f" % s.acceptance_fraction.mean())

        # 1d Marginals
        print("\nDetails for posterior one-dimensional marginals:")
        # Remove a sufficient number of burn-in steps
        burn = int(np.ceil(np.max(tau)) * 2)
        flat_samples = s.get_chain(discard=burn, flat=True)

        # FIGURES: Marginal posterior(s)
        print("\nPrinting PDF output")

        plotter(flat_samples[:, 0], 'alpha')
        plotter(flat_samples[:, 1], 'f0')
        plotter(flat_samples[:, 2], 'f1')

        # FIGURE: Joint posterior(s)
        corner.corner(
            flat_samples,
            labels=[r"$\alpha$", r"$f_0$", r"$f_1$"],
            quantiles=[0.025, 0.5, 0.975],
            show_titles=True,
            title_fmt=".3g",
            title_kwargs={"fontsize": 12}
        )
        plt.savefig('joint_post_corner.pdf', bbox_inches='tight')


def vis_data():
    data = read_data()
    plt.plot(data)
    plt.savefig('./tmp.png')


def problem1a(alpha, f):
    print(f"[bold]Problem 1a: For {alpha=}, {f=}, predicted T={likelihood.model_T_simple(alpha, f)}")


def lsqr_fn(x, data):
    """Least squares function to minimize."""
    alpha, f0, f1 = x
    model = likelihood.model_T(alpha, f0, f1, data["CO2"])
    return data["temperature"] - model


def problem1b():
    print("\n[bold]Problem 1b: Least squares optimization")
    data = read_data()
    init = (0, 0, 0)
    res: optimize.OptimizeResult = optimize.least_squares(
        lambda p: lsqr_fn(p, data),
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
    print(f"\tMSQR:", np.linalg.norm(res.fun) / len(data["temperature"]))
    f = likelihood.embedded_f_linear(f0, f1, data["CO2"])
    if np.any(f < 0) or np.any(f > 1):
        print(f"\t[yellow]Warning: {len(f[f > 1]) + len(f[f<0])} f values are out of bounds!")


if __name__ == '__main__':
    problem1a(0.3, 0.6)
    problem1b()
    pd.set_option('display.max_rows', None)
    # print(read_data().head(81))
    vis_data()
