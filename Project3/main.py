#!/bin/py
from matplotlib.ticker import FormatStrFormatter
from multiprocessing import Pool
from math import ceil, floor
from emcee import EnsembleSampler, autocorr
import likelihood
import prior
from scipy import stats
from matplotlib import pyplot as plt
import numpy as np
from cmcrameri import cm
import os
import corner

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
fdict = {'prior_p': prior.prior_p, 'prior_U': prior.prior_U, 'prior_C': prior.prior_C}

# -------------------------------------------------------------
# subroutine that generates a .pdf file plotting a quantity
# -------------------------------------------------------------


def plotter(chain, quant, xmin=None, xmax=None):
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


# -------------------------------------------------------------
# MCMC sampling Function
# -------------------------------------------------------------

class BayesianRichardsonExtrapolation():
    "Computes the Bayesian Richardson extrapolation posterior log density."

    def __call__(self, params, dtype=np.double):
        q, C, p = params

        return (
            prior.prior(q, C, p) +
            likelihood.likelihood(q, C, p)
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
    # -------------------------------------------------------------
    # Main Function
    # -------------------------------------------------------------
    # Example of sampling Bayesian Richardson extrapolation density using emcee

    #
    # initialize the Bayesian Calibration Procedure
    #

    print("\nInitializing walkers")
    nwalk = 100

    # initial guesses for the walkers starting locations
    guess_q = 1.16389876649
    guess_c = 0
    guess_p = 6

    params0 = np.tile([guess_q, guess_c, guess_p], nwalk).reshape(nwalk, 3)
    params0[:, 0] += np.random.randn(nwalk) * 0.025    # Perturb q
    params0[:, 1] += np.random.randn(nwalk) * 0.1      # Perturb C
    params0[:, 2] += np.random.randn(nwalk) * 1.5      # Perturb p...
    params0[:, 2] = np.absolute(params0[:, 2])        # ...and force >= 0

    print("\nInitializing the sampler and burning in walkers")
    with Pool(10) as pool:
        bre = BayesianRichardsonExtrapolation()
        s = EnsembleSampler(nwalk, params0.shape[-1], bre, pool=pool)
        pos, prob, state = s.run_mcmc(params0, 15000, progress=True)
        # tau = s.get_autocorr_time()
        # print(tau)
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

        #
        # 1d Marginals
        #
        print("\nDetails for posterior one-dimensional marginals:")
        # Remove a sufficient number of burn-in steps
        burn = int(np.ceil(np.max(tau)) * 2)
        flat_samples = s.get_chain(discard=burn, flat=True)

        # ----------------------------------
        # FIGURES: Marginal posterior(s)
        # ----------------------------------
        print("\nPrinting PDF output")

        plotter(flat_samples[:, 0], 'U')
        plotter(flat_samples[:, 1], 'C')
        plotter(flat_samples[:, 2], 'p')

        # ----------------------------------
        # FIGURE: Joint posterior(s)
        # ----------------------------------
        fig = corner.corner(
            flat_samples, labels=[
                "$q$", "$C$", "$p$"], quantiles=[0.025, 0.5, 0.975], show_titles=True, title_fmt=".3g", title_kwargs={"fontsize": 12})
        plt.savefig('joint_post_corner.pdf', bbox_inches='tight')


# Stop module loading when imported.  Otherwise continue running.
if __name__ == '__main__':
    main()
