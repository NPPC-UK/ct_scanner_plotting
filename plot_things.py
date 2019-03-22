from pathlib import Path
from math import pi
from statistics import mean

import numpy as np
import matplotlib.pyplot as plt

from plots import (plot_sorted_property, plot_property_vs_property, 
                   plot_pearson_correlations)

def get_nth_property(n, data):
    props = []
    names = []
    for i, v in data.items():
        props.append(v[0][:, n])
        names.append(i)

    return (props, names)

def get_properties(ns, data):
    d = []
    for i, v in data.items():
        name_item = (i,)
        for n in ns:
            name_item += (v[0][:, n],) 

        d.append(name_item)

    return d

def group_data(data, names):
    grouped_data = {}

    for d, name in zip(data, names):
        if name[:-3] not in grouped_data:
            grouped_data[name[:-3]] = list(d)
        else:
            grouped_data[name[:-3]] += list(d)

    return grouped_data

def filter_grains(grains, bottom, top):
    false_grains_idxs = []

    for idx, grain in enumerate(grains):
        bottom_dist = np.linalg.norm(grain[9:12] - bottom)
        top_dist = np.linalg.norm(grain[9:12] - top)

        near_ends = bottom_dist < 10 or top_dist < 10
        if near_ends:
            false_grains_idxs.append(idx)

    return np.delete(grains, false_grains_idxs, 0)


def get_data(meta_file, base_path):
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
        # matching.  I collect on the entire generator and retrieve the first
        # item.
        try:
            grains_file = list(csv_dir.glob('*.ISQ.csv'))[0]
        except IndexError:
            continue

        try: 
            length_file = list(csv_dir.glob('*.ISQlength.csv'))[0]
        except IndexError:
            continue

        length = np.genfromtxt(length_file, delimiter=',', skip_header=0)
        grains = filter_grains(
            np.genfromtxt(grains_file, delimiter=',', skip_header=1),
            length[1:4],
            length[4:]
        )

        data[scan[0]] = (grains, length[0])

    return data

def main():
    data = get_data(
        Path('/mnt/mass/max/BR09_CTdata/mnt/mass/scratch/br09_data/'
             'BR9_scan_list.csv'),
        Path('/mnt/mass/max/BR09_CTdata/mnt/mass/scratch/br09_data')
    )

    volumes, names = get_nth_property(5, data)
    plot_sorted_property([mean(volume) for volume in volumes], names, 
                         property_name='mean volume of grains')

    plot_sorted_property([mean(((3/(4*pi))*volume)**(1./3)) for volume in volumes], names,
                         property_name='mean adjusted radius of grains')

    n_grains = [len(grains) for grains in volumes]
    plot_sorted_property(n_grains, names, property_name='number of grains')

    grouped_volumes = group_data(volumes, names).items()

    names = [name for name, volume in grouped_volumes]
    volumes = [volume for name, volume in grouped_volumes]

    plot_sorted_property([mean(volume) for volume in volumes], names,
                         property_name='grouped mean volume of grains')

    n_grains = [len(grains) for grains in volumes]
    plot_sorted_property(n_grains, names, property_name='grouped number of grains')

    names, sphericities, lengths, n_grains, vols, surface_areas = (
        [], [], [], [], [], []
    )
    for name, (np_data, length) in data.items():
        names.append(name)

        ideal_surface_area = pi**(1./3)*(6*np_data[:, 5])**(2./3)
        sphericity = np.mean(np.divide(ideal_surface_area, np_data[:, 7]))
        sphericities.append(sphericity)

        lengths.append(length)

        n_grains.append(len(np_data[:, 0]))
        vols.append(np.mean(np_data[:, 5]))
        surface_areas.append(np.mean(np_data[:, 7]))

    plot_sorted_property(sphericities, names, 
                         property_name='mean sphericity of grains')
    plot_sorted_property(lengths, names, 
                         property_name='length of pod')

    plot_property_vs_property(lengths, n_grains, names, 'length of pod', 
                              'number of grains')

    plot_property_vs_property(lengths, vols, names, 'length of pod', 
                              'mean volume of grains')

    plot_property_vs_property(n_grains, vols, names, 'number of grains', 
                              'mean volume of grains')

    plot_property_vs_property(vols, sphericities, names, 'mean volume of grains',
                              'mean sphericities')

    plot_pearson_correlations(
        [lengths, n_grains, vols, sphericities, surface_areas],
        ['length', 'n_grains', 'volumes', 'sphericities', 'surface_areas']
    )

if __name__ == '__main__':
    main()
