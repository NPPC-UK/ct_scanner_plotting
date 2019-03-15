from pathlib import Path
from math import pi
from statistics import mean

import numpy as np

from plots import plot_sorted_property

def get_nth_property(n, data):
    props = []
    names = []
    for i, v in data.items():
        props.append(v[:, n])
        names.append(i)

    return (props, names)

def group_data(data, names):
    grouped_data = {}

    for d, name in zip(data, names):
        if name[:-2] not in grouped_data:
            grouped_data[name[:-2]] = list(d)
        else:
            grouped_data[name[:-2]] += list(d)

    return grouped_data

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
