from statistics import median
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.animation
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
import numpy as np
import numpy.polynomial.polynomial as poly
from pathos.multiprocessing import ProcessingPool

from scipy.stats import gaussian_kde
from scipy.stats.stats import pearsonr


pool = None

Axes3D  # Turn off flake8 warning about unused Axes3D


def _get_pool():
    global pool
    if pool is None:
        pool = ProcessingPool(3)

    return pool


def plot_swarm_property(containers, prop_fn, property_name="Property"):
    #    prop = _get_pool().map(prop_fn, containers)
    prop = list(map(prop_fn, containers))
    names = [con.name for con in containers]

    fig = plt.figure(1, figsize=(11, 8))
    axSwarm = fig.add_axes([0.05, 0.3, 0.949, 0.62], title=property_name)
    sns.swarmplot(data=prop, ax=axSwarm, size=1)

    axSwarm.set_xticks(range(0, len(names)))
    axSwarm.set_xticklabels(names, rotation=90, fontsize="small")

    return fig


def plot_bar_property(containers, prop_fn, property_name="Property"):
    prop = []
    #   prop = _get_pool().map(prop_fn, containers)
    prop = list(map(prop_fn, containers))

    names = [con.name for con in containers]

    fig = plt.figure(1, figsize=(11, 8))
    axBar = fig.add_axes([0.05, 0.3, 0.949, 0.62], title=property_name)
    axBar.boxplot(
        prop, bootstrap=1000, whis=3, flierprops=dict(marker=".", markersize=1)
    )

    axBar.set_xticks(range(1, len(names) + 1))
    axBar.set_xticklabels(names, rotation=90, fontsize="small")

    for i in range(1, len(names)):
        axBar.axvline(
            i + 0.5, color="black", linewidth=1, dashes=(0.05, 0.95), alpha=0.8
        )

    return fig


def plot_sorted_property(containers, prop_fn, property_name="Property"):
    #    prop = _get_pool().map(prop_fn, containers)
    prop = list(map(prop_fn, containers))
    names = [con.name for con in containers]
    median_prop = median(prop)

    sorted_data = sorted(zip(prop, names), key=lambda x: x[0])

    prop = [prop for prop, name in sorted_data]
    names = [name for prop, name in sorted_data]
    derivative_prop = [j - i for i, j in zip(prop[:-1], prop[1:])]

    derivative_prop_cp = list(derivative_prop)
    derivative_prop_cp[0:10] = [0] * 10
    derivative_prop_cp[-10:] = [0] * 10

    high_derivatives = sorted(
        enumerate(derivative_prop_cp), key=lambda x: x[1]
    )[-10:]

    left, width = 0.1, 0.65
    bottom, height = 0.35, 0.60
    left_h = left + width + 0.02

    fig = plt.figure(1, figsize=(16, 9))

    axScatter = fig.add_axes([left, bottom, width, height])

    axHist = fig.add_axes([left_h, bottom, 0.2, height], label="Histogram")
    axHist.yaxis.set_major_formatter(NullFormatter())

    axSmoothedHist = fig.add_axes(
        [left_h, bottom, 0.2, height], label="Smoothed Hist"
    )
    axSmoothedHist.yaxis.set_major_formatter(NullFormatter())
    axSmoothedHist.xaxis.set_major_formatter(NullFormatter())

    axDeriv = fig.add_axes([left, 0.05, width, 0.23])

    axScatter.scatter(
        range(0, len(prop)), prop, marker="+", s=50, linewidth=0.8
    )
    axScatter.plot(
        [0, len(prop)],
        [median_prop, median_prop],
        label="Median {}".format(property_name),
        linewidth=0.5,
        color="red",
    )

    axScatter.set_ylim(0, max(prop) * 1.1)

    n_bins = 40
    binwidth = (max(prop) - min(prop)) / n_bins
    bins = np.arange(0, max(prop) + binwidth, binwidth)

    density = gaussian_kde(prop)
    density.covariance_factor = lambda: 0.1
    density._compute_covariance()
    density_x = np.arange(0, max(prop), max(prop) / 500)
    smoothed_hist_data = density(density_x)
    axSmoothedHist.plot(
        smoothed_hist_data,
        density_x,
        color="red",
        linewidth=0.5,
        label="Smoothed Histogram",
    )
    axSmoothedHist.patch.set_visible(False)
    axSmoothedHist.set_ylim(axScatter.get_ylim())

    axHist.hist(prop, bins=bins, orientation="horizontal", label="Histogram")
    axHist.set_ylim(axScatter.get_ylim())
    axHist_xlim = axHist.get_xlim()

    axSmoothedHist.set_xlim([axHist_xlim[0], axHist_xlim[1]])

    axDeriv.plot(derivative_prop, color="black", linewidth=0.8)

    axDeriv.scatter(
        [i for i, v in high_derivatives],
        [v for i, v in high_derivatives],
        color="red",
        s=15,
    )

    axScatter.scatter(
        [i for i, v in high_derivatives],
        [prop[i] for i, v in high_derivatives],
        color="red",
        s=10,
    )

    t = (
        "Sorted highest Derivatives (red):\n" + ", ".join(["{}"] * 10) + "\n"
    ).format(*[names[i] for i, v in high_derivatives])

    t += ("Lowest Values:\n" + ", ".join(["{}"] * 10) + "\n").format(
        *names[:10]
    )

    t += ("Highest Values:\n" + ", ".join(["{}"] * 10) + "\n").format(
        *names[-10::-1]
    )

    fig.text(left_h, 0.05, t, wrap=True, fontsize=9)

    axScatter.set_title("Numerically sorted {}".format(property_name))

    axHist.legend(loc="upper right")
    axSmoothedHist.legend(loc="lower right")

    return fig


def plot_property_vs_property(
    containers,
    x_prop_fn,
    y_prop_fn,
    x_prop_name="X Prop",
    y_prop_name="Y Prop",
):
    # x_prop = _get_pool().map(x_prop_fn, containers)
    # y_prop = _get_pool().map(y_prop_fn, containers)
    x_prop = list(map(x_prop_fn, containers))
    y_prop = list(map(y_prop_fn, containers))
    left, width = 0.1, 0.65
    bottom, height = 0.35, 0.60

    fig = plt.figure(1, figsize=(16, 9))
    axScatter = fig.add_axes(
        [left, bottom, width, height],
        label="{} vs {}".format(x_prop_name, y_prop_name),
        xlabel=x_prop_name,
        ylabel=y_prop_name,
    )
    axScatter.scatter(x_prop, y_prop, marker="+", s=50)
    axScatter.set_title("{} vs {}".format(x_prop_name, y_prop_name))

    return fig


def plot_pearson_correlations(containers, props_fns, prop_names):
    props = []
    for fn in props_fns:
        #   props.append(_get_pool().map(fn, containers))
        props.append(list(map(fn, containers)))

    correlations = np.zeros((len(props), len(props)))
    p_values = np.ones((len(props), len(props)))

    for i, j in combinations(range(len(props)), 2):
        ith = np.array(props[i])
        jth = np.array(props[j])
        bad_idx = ~np.logical_or(np.isnan(ith), np.isnan(jth))
        pcc, p = pearsonr(np.compress(bad_idx, ith), np.compress(bad_idx, jth))
        correlations[i, j] = pcc
        p_values[i, j] = p

    np.unique(correlations.ravel())

    left, width = 0.1, 0.65
    bottom, height = 0.35, 0.60

    fig = plt.figure(1, figsize=(16, 9))

    axHeat = fig.add_axes([left, bottom, width, height])

    axHeat.imshow(correlations)
    axHeat.set_xticks(np.arange(len(prop_names)))
    axHeat.set_yticks(np.arange(len(prop_names)))
    axHeat.set_xticklabels(prop_names)
    axHeat.set_yticklabels(prop_names)

    plt.setp(
        axHeat.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    for i in range(len(props)):
        for j in range(len(props)):
            axHeat.text(
                j,
                i,
                "{0:.2f}, {1:.2f}".format(correlations[i, j], p_values[i, j]),
                ha="center",
                va="center",
                color="w",
                fontsize=8,
            )

    return fig


def plot_spine_debug(pod, name):
    xs = [s.position.x for s in pod.slices]
    ys = [s.position.y for s in pod.slices]
    zs = [s.position.z for s in pod.slices]

    x_ffit = poly.Polynomial(pod.spine[0])
    y_ffit = poly.Polynomial(pod.spine[1])
    x_new = np.linspace(zs[0], zs[-1])

    fig = plt.figure(figsize=[1, 4.8])
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x_ffit(x_new), y_ffit(x_new), x_new)
    ax.plot(xs, ys, zs, linewidth=0.5)
    ax.plot(
        [s.position.x for s in pod.seeds],
        [s.position.y for s in pod.seeds],
        [s.position.z for s in pod.seeds],
        ms=3,
        linestyle="",
        markerfacecolor="None",
        markeredgecolor="green",
        alpha=0.6,
        marker="o",
    )

    ax.view_init(10, 0)
    phi = np.linspace(0, 2 * np.pi, num=200)

    def update(phi):
        ax.view_init(10, phi * 180.0 / np.pi)

    ani = matplotlib.animation.FuncAnimation(fig, update, frames=phi)
    ani.save("plot_{}.gif".format(name), fps=18, dpi=196, writer="imagemagick")


def plot_kde_debug(genotype, name):
    score, xs = genotype.kde()
    plt.plot(xs, score)
    plt.savefig("plot_{}.svg".format(name))
    plt.clf()
