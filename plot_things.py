from pathlib import Path
from statistics import median, mean
from math import pi

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from scipy.stats import gaussian_kde

meta_file = '/mnt/mass/max/BR09_CTdata/mnt/mass/scratch/br09_data/BR9_scan_list.csv'
base_path = Path('/mnt/mass/max/BR09_CTdata/mnt/mass/scratch/br09_data')

meta_type = np.dtype([
    ('sample_name', np.unicode_, 12),
    ('folder', np.unicode_, 8)
])


meta_data = np.genfromtxt(meta_file, delimiter='\t', usecols=[0, 4],
                          dtype=meta_type, skip_header=1)

data = {}

for scan in meta_data:
    csv_dir = base_path/scan[1]

    # Glob returns a generator.  I know that there is only one file
    # matchin.  I collect on the entire generator and retrieve the first
    # item.
    try:
        grains_file = list(csv_dir.glob('*.ISQ.csv'))[0]
    except IndexError:
        continue

    grains = np.genfromtxt(grains_file, delimiter=',', skip_header=1)

    data[scan[0]] = grains


def get_nth_property(n, data):
    props = []
    names = []
    for i, v in data.items():
        props.append(v[:, n])
        names.append(i)

    return (props, names)


def plot_sorted_property(prop, names, property_name='Property'):
    median_prop = median(prop)

    prop = sorted(prop) 
    derivative_prop = [j-i for i, j in zip(prop[:-1], prop[1:])]

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
    
    axSmoothedHist.set_xlim([axHist.get_xlim()[0], axSmoothedHist.get_xlim()[1]])

    axDeriv.plot(derivative_prop, color='black', linewidth=0.8)

    fig.savefig('plot_{}.svg'.format(property_name.replace(' ', '_')))
    fig.clf()


volumes, names = get_nth_property(5, data)
plot_sorted_property([mean(volume) for volume in volumes], names, 
              property_name='mean volume of grains')

plot_sorted_property([mean(((3/(4*pi))*volume)**(1./3)) for volume in volumes], names,
              property_name='mean adjusted radius of grains')

n_grains = [len(grains) for grains in volumes]
plot_sorted_property(n_grains, names, property_name='number of grains')


