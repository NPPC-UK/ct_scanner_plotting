from statistics import median

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

import numpy as np

from scipy.stats import gaussian_kde


def plot_sorted_property(prop, names, property_name='Property'):
    median_prop = median(prop)

    sorted_data = sorted(zip(prop, names), key=lambda x: x[0]) 

    prop = [prop for prop, name in sorted_data]
    derivative_prop = [j-i for i, j in zip(prop[:-1], prop[1:])]

    derivative_prop_cp = list(derivative_prop)
    derivative_prop_cp[0:10] = [0] * 10
    derivative_prop_cp[-10:] = [0] * 10

    high_derivatives = sorted(enumerate(derivative_prop_cp), 
                              key=lambda x: x[1])[-10:]

    left, width = 0.1, 0.65
    bottom, height = 0.35, 0.64
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
    axSmoothedHist.plot(smoothed_hist_data, density_x, color='red', linewidth=0.5)
    axSmoothedHist.patch.set_visible(False)
    axSmoothedHist.set_ylim(axScatter.get_ylim())

    axHist.hist(prop, bins=bins, orientation='horizontal')
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


    fig.savefig('plot_{}.svg'.format(property_name.replace(' ', '_')))
    fig.clf()
