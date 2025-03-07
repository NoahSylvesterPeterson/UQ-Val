#!/bin/py
from matplotlib.ticker import FormatStrFormatter
from multiprocessing import Pool
from math import ceil, floor
from emcee import EnsembleSampler
import likelihood
import prior
from scipy import stats
import pylab
from matplotlib import pyplot as plt
import numpy as np
from cmcrameri import cm

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
    params0.T[0] += np.random.rand(nwalk) * 0.025    # Perturb q
    params0.T[1] += np.random.rand(nwalk) * 0.1      # Perturb C
    params0.T[2] += np.random.rand(nwalk) * 1.5      # Perturb p...
    params0.T[2] = np.absolute(params0.T[2])        # ...and force >= 0

    print("\nInitializing the sampler and burning in walkers")
    with Pool(8) as pool:
        s = EnsembleSampler(nwalk, params0.shape[-1], bre, pool=pool)
        pos, prob, state = s.run_mcmc(params0, 5000, progress=True)
        s.reset()
        print("\nSampling the posterior density for the problem")
        s.run_mcmc(pos, 10000, progress=True)

        print("Mean acceptance fraction was %.3f" % s.acceptance_fraction.mean())

        #
        # 1d Marginals
        #
        print("\nDetails for posterior one-dimensional marginals:")

        qm, qs = textual_boxplot("q", s.flatchain[:, 0], header=True)
        cm, cs = textual_boxplot("C", s.flatchain[:, 1], header=False)
        pm, ps = textual_boxplot("p", s.flatchain[:, 2], header=False)

        # ----------------------------------
        # FIGURES: Marginal posterior(s)
        # ----------------------------------
        print("\nPrinting PDF output")

        plotter(s.flatchain[:, 0], 'U')
        plotter(s.flatchain[:, 1], 'C')
        plotter(s.flatchain[:, 2], 'p')

        # ----------------------------------
        # FIGURE: Joint posterior(s)
        # ----------------------------------

        qbins = np.linspace(np.min(s.flatchain[:, 0]), np.max(s.flatchain[:, 0]), 200)
        Cbins = np.linspace(np.min(s.flatchain[:, 1]), np.max(s.flatchain[:, 1]), 200)
        pbins = np.linspace(np.min(s.flatchain[:, 2]), np.max(s.flatchain[:, 2]), 200)

        qkde = stats.gaussian_kde(s.flatchain[:, 0])
        Ckde = stats.gaussian_kde(s.flatchain[:, 1])
        pkde = stats.gaussian_kde(s.flatchain[:, 2])

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

        plt.figure()

        formatter = FormatStrFormatter('%5.4f')
        formatter2 = FormatStrFormatter('%5.f')

        pylab.subplot(3, 3, 1)
        plt.plot(qbins, qpdf, linewidth=2, color="k", label="Post")

        plt.xlim(qbounds)
        pylab.gca().set_xticks(qticks)
        pylab.gca().xaxis.set_major_formatter(formatter)
        pylab.gca().xaxis.set_minor_formatter(formatter)
        pylab.gca().set_yticks([])
        plt.xlabel('$q$', fontsize=24)

        pylab.subplot(3, 3, 2)
        H, qe, Ce = np.histogram2d(s.flatchain[:, 0], s.flatchain[:, 1], bins=(200, 200))

        qv = 0.5 * (qe[0:-1] + qe[1:len(qe)])
        Cv = 0.5 * (Ce[0:-1] + Ce[1:len(Ce)])

        plt.contour(Cv, qv, H, 5, colors='k')

        plt.xlim(Cbounds)
        pylab.gca().set_xticks(Cticks)
        pylab.gca().set_xticklabels([])

        # plt.ylim(qbounds)
        pylab.gca().set_yticks(qticks)
        pylab.gca().set_yticklabels([])

        pylab.subplot(3, 3, 3)
        H, qe, pe = np.histogram2d(s.flatchain[:, 0], s.flatchain[:, 2], bins=(200, 200))

        qv = 0.5 * (qe[0:-1] + qe[1:len(qe)])
        pv = 0.5 * (pe[0:-1] + pe[1:len(pe)])

        plt.contour(pv, qv, H, 5, colors='k')

        plt.xlim(pbounds)
        pylab.gca().set_xticks(pticks)
        pylab.gca().set_xticklabels([])

        plt.ylim(qbounds)
        pylab.gca().set_yticks(qticks)
        pylab.gca().set_yticklabels([])

        pylab.subplot(3, 3, 5)
        plt.plot(Cbins, Cpdf, linewidth=2, color="k", label="Post")
        pylab.gca().xaxis.set_major_formatter(formatter)
        pylab.gca().xaxis.set_minor_formatter(formatter)
        pylab.gca().set_yticks([])
        plt.xlabel('$C$', fontsize=24)

        plt.xlim(Cbounds)
        pylab.gca().set_xticks(Cticks)

        pylab.subplot(3, 3, 6)
        H, Ce, pe = np.histogram2d(s.flatchain[:, 1], s.flatchain[:, 2], bins=(200, 200))

        Cv = 0.5 * (Ce[0:-1] + Ce[1:len(Ce)])
        pv = 0.5 * (pe[0:-1] + pe[1:len(pe)])

        plt.contour(pv, Cv, H, 5, colors='k')

        plt.xlim(pbounds)
        pylab.gca().set_xticks(pticks)
        pylab.gca().set_xticklabels([])

        plt.ylim(Cbounds)
        pylab.gca().set_yticks(Cticks)
        pylab.gca().set_yticklabels([])

        pylab.subplot(3, 3, 9)
        plt.plot(pbins, ppdf, linewidth=2, color="k", label="Post")
        pylab.gca().xaxis.set_major_formatter(formatter2)
        pylab.gca().xaxis.set_minor_formatter(formatter2)
        pylab.gca().set_yticks([])
        plt.xlabel('$p$', fontsize=24)

        plt.xlim(pbounds)
        pylab.gca().set_xticks(pticks)
        plt.savefig('joint_post.pdf', bbox_inches='tight')


# Stop module loading when imported.  Otherwise continue running.
if __name__ == '__main__':
    main()
