#!/bin/py
from matplotlib.ticker import FormatStrFormatter
from multiprocessing import Pool
from math import ceil, floor
from emcee import EnsembleSampler
import likelihood
import prior
from scipy import stats
from matplotlib import pyplot as plt
import numpy as np
from cmcrameri import cm
import os

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
    bre = BayesianRichardsonExtrapolation()

    print("\nInitializing walkers")
    nwalk = 100

    # initial guesses for the walkers starting locations
    guess_q = 1.16389876649
    guess_c = 0
    guess_p = 6

    params0 = np.tile([guess_q, guess_c, guess_p], nwalk).reshape(nwalk, 3)
    params0.T[0] += np.random.randn(nwalk) * 0.025    # Perturb q
    params0.T[1] += np.random.randn(nwalk) * 0.1      # Perturb C
    params0.T[2] += np.random.randn(nwalk) * 1.5      # Perturb p...
    params0.T[2] = np.absolute(params0.T[2])        # ...and force >= 0

    print("\nInitializing the sampler and burning in walkers")
    with Pool(10) as pool:
        s = EnsembleSampler(nwalk, params0.shape[-1], bre, pool=pool)
        pos, prob, state = s.run_mcmc(params0, 5000, progress=True)
        s.reset()
        print("\nSampling the posterior density for the problem")
        s.run_mcmc(pos, 100000, progress=True)

        print("Mean acceptance fraction was %.3f" % s.acceptance_fraction.mean())

        #
        # 1d Marginals
        #
        print("\nDetails for posterior one-dimensional marginals:")
        flat_samples = s.get_chain(discard=150, thin=10, flat=True)

        qm, qs = textual_boxplot("q", flat_samples[:, 0], header=True)
        cm, cs = textual_boxplot("C", flat_samples[:, 1], header=False)
        pm, ps = textual_boxplot("p", flat_samples[:, 2], header=False)

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
        _, axes = plt.subplots(3, 3, figsize=(6, 6))

        qbins = np.linspace(np.min(flat_samples[:, 0]), np.max(flat_samples[:, 0]), 200)
        Cbins = np.linspace(np.min(flat_samples[:, 1]), np.max(flat_samples[:, 1]), 200)
        pbins = np.linspace(np.min(flat_samples[:, 2]), np.max(flat_samples[:, 2]), 200)

        qkde = stats.gaussian_kde(flat_samples[:, 0])
        Ckde = stats.gaussian_kde(flat_samples[:, 1])
        pkde = stats.gaussian_kde(flat_samples[:, 2])

        qpdf = qkde.evaluate(qbins)
        Cpdf = Ckde.evaluate(Cbins)
        ppdf = pkde.evaluate(pbins)

        # TODO FIX ME
        qbounds = np.array([qm - qs, qm + qs])
        Cbounds = np.array([cm - cs, cm + cs])
        pbounds = np.array([pm - ps, pm + ps])

        qticks = np.linspace(qbounds[0], qbounds[1], 3)
        Cticks = np.linspace(Cbounds[0], Cbounds[1], 3)
        pticks = np.linspace(pbounds[0], pbounds[1], 5)

        formatter = FormatStrFormatter('%5.4f')
        formatter2 = FormatStrFormatter('%5.f')

        axes[1, 0].axis('off')
        axes[2, 0].axis('off')
        axes[2, 1].axis('off')

        axes[0, 0].plot(qbins, qpdf, linewidth=2, color="k", label="Post")

        axes[0, 0].set_xlim(qbounds)
        axes[0, 0].set_xticks(qticks)
        axes[0, 0].xaxis.set_major_formatter(formatter)
        axes[0, 0].xaxis.set_minor_formatter(formatter)
        axes[0, 0].set_yticks([])
        axes[0, 0].set_xlabel('$q$', fontsize=24)

        H, qe, Ce = np.histogram2d(flat_samples[:, 0], flat_samples[:, 1], bins=(200, 200))

        qv = 0.5 * (qe[0:-1] + qe[1:len(qe)])
        Cv = 0.5 * (Ce[0:-1] + Ce[1:len(Ce)])

        axes[0, 1].contour(Cv, qv, H, 5, colors='k')

        axes[0, 1].set_xlim(Cbounds)
        axes[0, 1].set_xticks(Cticks)
        axes[0, 1].set_xticklabels([])

        # plt.ylim(qbounds)
        axes[0, 1].set_yticks(qticks)
        axes[0, 1].set_yticklabels([])

        H, qe, pe = np.histogram2d(flat_samples[:, 0], flat_samples[:, 2], bins=(200, 200))

        qv = 0.5 * (qe[0:-1] + qe[1:len(qe)])
        pv = 0.5 * (pe[0:-1] + pe[1:len(pe)])

        axes[0, 2].contour(pv, qv, H, 5, colors='k')

        axes[0, 2].set_xlim(pbounds)
        axes[0, 2].set_xticks(pticks)
        axes[0, 2].set_xticklabels([])

        axes[0, 2].set_ylim(qbounds)
        axes[0, 2].set_yticks(qticks)
        axes[0, 2].set_yticklabels([])

        axes[1, 1].plot(Cbins, Cpdf, linewidth=2, color="k", label="Post")
        axes[1, 1].xaxis.set_major_formatter(formatter)
        axes[1, 1].xaxis.set_minor_formatter(formatter)
        axes[1, 1].set_yticks([])
        axes[1, 1].set_xlabel('$C$', fontsize=24)

        axes[1, 1].set_xlim(Cbounds)
        axes[1, 1].set_xticks(Cticks)

        H, Ce, pe = np.histogram2d(flat_samples[:, 1], flat_samples[:, 2], bins=(200, 200))

        Cv = 0.5 * (Ce[0:-1] + Ce[1:len(Ce)])
        pv = 0.5 * (pe[0:-1] + pe[1:len(pe)])

        axes[1, 2].contour(pv, Cv, H, 5, colors='k')

        axes[1, 2].set_xlim(pbounds)
        axes[1, 2].set_xticks(pticks)
        axes[1, 2].set_xticklabels([])

        axes[1, 2].set_ylim(Cbounds)
        axes[1, 2].set_yticks(Cticks)
        axes[1, 2].set_yticklabels([])

        axes[2, 2].plot(pbins, ppdf, linewidth=2, color="k", label="Post")
        axes[2, 2].xaxis.set_major_formatter(formatter2)
        axes[2, 2].xaxis.set_minor_formatter(formatter2)
        axes[2, 2].set_yticks([])
        axes[2, 2].set_xlabel('$p$', fontsize=24)

        axes[2, 2].set_xlim(pbounds)
        axes[2, 2].set_xticks(pticks)
        plt.savefig('joint_post.pdf', bbox_inches='tight')


# Stop module loading when imported.  Otherwise continue running.
if __name__ == '__main__':
    main()
