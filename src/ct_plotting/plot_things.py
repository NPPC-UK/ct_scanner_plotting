from pathlib import Path
from math import pi
from statistics import mean

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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

def merge_grains(grains):
    """Return a grain corresponding to the 'sum' of an arbitrary number of
    other grains.

    Many parameters are meaningless of a merged grain.  Only the center of
    mass, and total volume make complete sense.  Surface area is more iffy.
    Should we count the surface of the grain fragments that are facing each
    other? The simple sum is calculated here. 

    Return a merged grains that has null values except where explained above.
    """
    xc, yc, zc = 0, 0, 0

    xc = (sum([grain[5]*grain[9] for grain in grains])/
          sum([grain[5] for grain in grains]))

    yc = (sum([grain[5]*grain[10] for grain in grains])/
          sum([grain[5] for grain in grains]))

    zc = (sum([grain[5]*grain[11] for grain in grains])/
          sum([grain[5] for grain in grains]))
    return np.array([
        -1, 
        -1, 
        -1, 
        -1, 
        -1, 
        sum([grain[5] for grain in grains]), 
        -1, 
        sum([grain[7] for grain in grains]),
        -1,
        xc,
        yc,
        zc,
        1,
        1
    ])


def filter_grains(grains, bottom, top):
    """Filter grains of a pod.

    Delete those grains that are too close to the ends of a pod to be real.
    Merge those whose centres are too close together to be real.

    Return the rest.
    """
    false_grains_idxs = []

    for idx, grain in enumerate(grains):
        bottom_dist = np.linalg.norm(grain[9:12] - bottom)
        top_dist = np.linalg.norm(grain[9:12] - top)

        near_ends = bottom_dist < 100 or top_dist < 100
        if near_ends:
            false_grains_idxs.append(idx)

    filtered_grains = np.delete(grains, false_grains_idxs, 0)

    grain_merge_candidates = []

    for idx, grain in enumerate(filtered_grains):
        distances = np.linalg.norm(
            filtered_grains[:, 9:12] - grain[9:12],
            axis=1
        )
        for dist_idx, dist in enumerate(distances):
            if dist_idx <= idx:
                continue

            if dist < 10:
                grain_merge_candidates.append((idx, dist_idx))

    G = nx.Graph()
    G.add_edges_from(grain_merge_candidates)
    to_delete = []
    for to_merge in nx.connected_components(G): 
        filtered_grains += merge_grains([filtered_grains[i] for i in to_merge])
        to_delete += to_merge

    return np.delete(filtered_grains, to_delete, 0)


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
        grains = np.genfromtxt(grains_file, delimiter=',', skip_header=1)

        filtered_grains = filter_grains(
            grains,
            length[1:4],
            length[4:]
        )

        data[scan[0]] = (grains, length[0], filtered_grains)

    return data

def main():
    data = get_data(
        Path('/mnt/mass/max/BR09_CTdata/mnt/mass/scratch/br09_data/'
             'BR9_scan_list.csv'),
        Path('/mnt/mass/max/BR09_CTdata/mnt/mass/scratch/br09_data')
    )


    volumes, names = get_nth_property(5, data)
    plot_sorted_property(
        [mean(volume) if len(volume) > 0 else -1 for volume in volumes], 
        names, property_name='mean volume of grains'
    )

    plot_sorted_property(
        [mean(((3/(4*pi))*volume)**(1./3)) if len(volume) > 0 else -1 
         for volume in volumes], 
        names, property_name='mean adjusted radius of grains'
    )

    n_grains = [len(grains) for grains in volumes]
    plot_sorted_property(n_grains, names, property_name='number of grains')

    grouped_volumes = group_data(volumes, names).items()

    names = [name for name, volume in grouped_volumes]
    volumes = [volume for name, volume in grouped_volumes]

    plot_sorted_property(
        [mean(volume) if len(volume) > 0 else -1 for volume in volumes], 
        names, property_name='grouped mean volume of grains'
    )

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

def plot_distances():
    all_data = get_data(
        Path('/mnt/mass/max/BR09_CTdata/mnt/mass/scratch/br09_data/BR9_scan_list.csv'),
        Path('/mnt/mass/max/BR09_CTdata/mnt/mass/scratch/br09_data/'),
    )

    distances = {}
    for name, data in all_data.items():
        distances[name] = []
        coords = data[0][:, 9:12][data[0][:, 11].argsort()]
        for i in range(1, len(coords[:, 0])):
            d = np.linalg.norm(coords[i, :] - coords[i - 1, :])
            # Ignore very small separations.  These are not separate grains
            if d > 5:
                distances[name].append(d)

    grouped_distances = {} 
    for name, data in distances.items():
        if name[:-3] not in grouped_distances:
            grouped_distances[name[:-3]] = distances[name]
        else:
            grouped_distances[name[:-3]] += distances[name]

        if name[:-3] == 'BR9_09911':
            print(name, sorted(data)[::-1])

    names = []
    distances = []
    max_distance = 0
    
    for key, value in grouped_distances.items():
        names.append(key)
        distances.append(value)

        max_distance = max(value) if max(value) > max_distance else max_distance

    for i in range(0, len(distances), 10):
        start = i
        end = i + 10 if len(distances) > i + 10 else len(distances)
        plt.violinplot(distances[start:end], showmeans=True, widths=0.9)
        plt.xticks(range(1, (end - start) + 1), 
                   names[start:end], rotation='vertical')
        plt.ylim(0, max_distance + 10)
        plt.savefig('plot_distances_between_grains_{}.svg'.format(i/10))
        plt.clf()

    return grouped_distances

if __name__ == '__main__':
    main()
