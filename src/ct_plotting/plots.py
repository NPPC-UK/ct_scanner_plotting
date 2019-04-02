from statistics import median
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

import numpy as np

from scipy.stats import gaussian_kde
from scipy.stats.stats import pearsonr


def plot_sorted_property(prop, names, property_name='Property'):
    median_prop = median(prop)

    sorted_data = sorted(zip(prop, names), key=lambda x: x[0]) 

    prop = [prop for prop, name in sorted_data]
    names = [name for prop, name in sorted_data]
    derivative_prop = [j-i for i, j in zip(prop[:-1], prop[1:])]

    derivative_prop_cp = list(derivative_prop)
    derivative_prop_cp[0:10] = [0] * 10
    derivative_prop_cp[-10:] = [0] * 10

    high_derivatives = sorted(enumerate(derivative_prop_cp), 
                              key=lambda x: x[1])[-10:]

    left, width = 0.1, 0.65
    bottom, height = 0.35, 0.60
    left_h = left + width + 0.02

    fig = plt.figure(1, figsize=(16, 9))

    axScatter = fig.add_axes([left, bottom, width, height])

    axHist = fig.add_axes([left_h, bottom, 0.2, height],
                          label='Histogram')
    axHist.yaxis.set_major_formatter(NullFormatter())

    axSmoothedHist = fig.add_axes([left_h, bottom, 0.2, height], 
                                  label='Smoothed Hist')
    axSmoothedHist.yaxis.set_major_formatter(NullFormatter())
    axSmoothedHist.xaxis.set_major_formatter(NullFormatter())

    axDeriv = fig.add_axes([left, 0.05, width, 0.23])

    axScatter.scatter(range(0, len(prop)), prop, marker="+", s=50,
                      linewidth=0.8)
    axScatter.plot([0, len(prop)], [median_prop, median_prop],
                   label='Median {}'.format(property_name), 
                   linewidth=0.5, color='red')

    axScatter.set_ylim(0, max(prop)*1.1)


    n_bins = 40
    binwidth = (max(prop) - min(prop))/n_bins
    bins = np.arange(0, max(prop) + binwidth, binwidth)

    density = gaussian_kde(prop)
    density.covariance_factor = lambda : .1
    density._compute_covariance()
    density_x = np.arange(0, max(prop), max(prop)/500)
    smoothed_hist_data = density(density_x)
    axSmoothedHist.plot(smoothed_hist_data, density_x, color='red', linewidth=0.5, label='Smoothed Histogram')
    axSmoothedHist.patch.set_visible(False)
    axSmoothedHist.set_ylim(axScatter.get_ylim())

    axHist.hist(prop, bins=bins, orientation='horizontal', label='Histogram')
    axHist.set_ylim(axScatter.get_ylim())
    axHist_xlim = axHist.get_xlim()

    axSmoothedHist.set_xlim([axHist.get_xlim()[0], axSmoothedHist.get_xlim()[1]])

    axDeriv.plot(derivative_prop, color='black', linewidth=0.8)

    axDeriv.scatter([i for i, v in high_derivatives], 
                    [v for i, v in high_derivatives],
                    color='red', s=15) 

    axScatter.scatter([i for i, v in high_derivatives],
                      [prop[i] for i, v in high_derivatives],
                      color='red', s=10)

    t = ("Sorted highest Derivatives (red):\n" + 
         ", ".join(["{}"]*10) + 
         "\n").format(*[names[i] for i, v in high_derivatives])

    t += ("Lowest Values:\n" +
          ", ".join(["{}"]*10) + 
          "\n").format(*names[:10])

    t += ("Highest Values:\n" +
          ", ".join(["{}"]*10) + 
          "\n").format(*names[-10::-1])

    fig.text(left_h, 0.05, t, wrap=True, fontsize=9)

    axScatter.set_title('Numerically sorted {}'.format(property_name))

    axHist.legend(loc='upper right')
    axSmoothedHist.legend(loc='lower right')

    fig.savefig('plot_{}.svg'.format(property_name.replace(' ', '_')))
    fig.clf()

def plot_property_vs_property(x_prop, y_prop, names, 
                              x_prop_name='X Prop',
                              y_prop_name='Y Prop'):
    left, width = 0.1, 0.65
    bottom, height = 0.35, 0.60
    left_h = left + width + 0.02

    fig = plt.figure(1, figsize=(16, 9))
    axScatter = fig.add_axes([left, bottom, width, height], 
                             label="{} vs {}".format(x_prop_name, y_prop_name),
                             xlabel=x_prop_name, ylabel=y_prop_name)
    axScatter.scatter(x_prop, y_prop, marker='+', s=50) 
    axScatter.set_title('{} vs {}'.format(x_prop_name, y_prop_name))


    fig.savefig('plot_{}_vs_{}.svg'.format(
        x_prop_name.replace(' ', '_'),
        y_prop_name.replace(' ', '_')
    ))
    fig.clf()

def plot_pearson_correlations(props, prop_names):
    correlations = np.zeros((len(props), len(props)))
    p_values = np.ones((len(props), len(props)))

    for i, j in combinations(range(len(props)), 2):
        pcc, p = pearsonr(props[i], props[j])
        correlations[i, j] = pcc
        p_values[i, j] = p

    values = np.unique(correlations.ravel())

    left, width = 0.1, 0.65
    bottom, height = 0.35, 0.60
    left_h = left + width + 0.02

    fig = plt.figure(1, figsize=(16, 9))


    axHeat = fig.add_axes([left, bottom, width, height])

    im = axHeat.imshow(correlations)
    axHeat.set_xticks(np.arange(len(prop_names)))
    axHeat.set_yticks(np.arange(len(prop_names)))
    axHeat.set_xticklabels(prop_names)
    axHeat.set_yticklabels(prop_names)

    plt.setp(axHeat.get_xticklabels(), rotation=45, 
            ha='right', rotation_mode='anchor')

    for i in range(len(props)):
        for j in range(len(props)):
            text = axHeat.text(
                j, i, 
                "{0:.2f}, {1:.2f}".format(correlations[i, j], p_values[i, j]),
                ha="center", va="center", color="w"
            )



    fig.savefig('plot_correlations.svg') 
    fig.clf()