from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from ct_plotting.plots import (
    plot_sorted_property,
    plot_property_vs_property,
    plot_pearson_correlations,
)

from ct_plotting.data import Pod, Plant


def group_data(pods):
    grouped = {}

    for pod in pods:
        if pod.name[:-3] not in grouped:
            grouped[pod.name[:-3]] = [pod]
        else:
            grouped[pod.name[:-3]].append(pod)

    plants = []
    for name, pods in grouped.items():
        plants.append(Plant(pods, name))

    return plants


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

    xc = sum([grain[5] * grain[9] for grain in grains]) / sum(
        [grain[5] for grain in grains]
    )

    yc = sum([grain[5] * grain[10] for grain in grains]) / sum(
        [grain[5] for grain in grains]
    )

    zc = sum([grain[5] * grain[11] for grain in grains]) / sum(
        [grain[5] for grain in grains]
    )
    return np.array(
        [
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
            1,
        ]
    )


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
            filtered_grains[:, 9:12] - grain[9:12], axis=1
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
    meta_type = np.dtype(
        [("sample_name", np.unicode_, 12), ("folder", np.unicode_, 8)]
    )

    meta_data = np.genfromtxt(
        meta_file,
        delimiter="\t",
        usecols=[0, 4, 6],
        dtype=meta_type,
        skip_header=1,
    )

    pods = []

    for scan in meta_data:
        csv_dir = base_path / scan[1]

        # Glob returns a generator.  I know that there is only one file
        # matching.  I collect on the entire generator and retrieve the first
        # item.
        try:
            grains_file = list(csv_dir.glob("*.ISQ.csv"))[0]
        except IndexError:
            continue

        try:
            length_file = list(csv_dir.glob("*.ISQlength.csv"))[0]
        except IndexError:
            continue

        pod = Pod.pod_from_files(Pod, grains_file, length_file, scan[0])

        pods.append(pod)

    return pods


def main():
    pods = get_data(
        Path(
            "/mnt/mass/max/BR09_CTdata/mnt/mass/scratch/br09_data/"
            "BR9_scan_list.csv"
        ),
        Path("/mnt/mass/max/BR09_CTdata/mnt/mass/scratch/br09_data"),
    )
    plants = group_data(pods)

    plot_sorted_property(
        pods,
        lambda pods: [pod.mean_volume() for pod in pods],
        property_name="mean volume of grains",
    )

    plot_sorted_property(
        pods,
        lambda pods: [pod.n_grains() for pod in pods],
        property_name="number of grains",
    )

    plot_sorted_property(
        plants,
        lambda plants: [plant.mean_volume() for plant in plants],
        property_name="grouped mean volume of grains",
    )

    plot_sorted_property(
        plants,
        lambda plants: [plant.n_grains() for plant in plants],
        property_name="grouped number of grains",
    )

    plot_sorted_property(
        pods,
        lambda pods: [pod.mean_sphericity() for pod in pods],
        property_name="mean sphericity of grains",
    )

    plot_sorted_property(
        pods,
        lambda pods: [pod.length() for pod in pods],
        property_name="length of pod",
    )

    plot_property_vs_property(
        pods,
        lambda pods: [pod.length() for pod in pods],
        lambda pods: [pod.n_grains() for pod in pods],
        "length of pod",
        "number of grains",
    )

    plot_property_vs_property(
        pods,
        lambda pods: [pod.length() for pod in pods],
        lambda pods: [pod.mean_volume() for pod in pods],
        "length of pod",
        "mean volume of grains",
    )

    plot_property_vs_property(
        pods,
        lambda pods: [pod.n_grains() for pod in pods],
        lambda pods: [pod.mean_volume() for pod in pods],
        "number of grains",
        "mean volume of grains",
    )

    plot_property_vs_property(
        pods,
        lambda pods: [pod.mean_volume() for pod in pods],
        lambda pods: [pod.mean_sphericity() for pod in pods],
        "mean volume of grains",
        "mean sphericities",
    )

    plot_pearson_correlations(
        pods,
        [
            lambda pods: [pod.length() for pod in pods],
            lambda pods: [pod.n_grains() for pod in pods],
            lambda pods: [pod.mean_volume() for pod in pods],
            lambda pods: [pod.mean_sphericity() for pod in pods],
            lambda pods: [pod.mean_surface_area() for pod in pods],
        ],
        ["length", "n_grains", "volumes", "sphericities", "surface_areas"],
    )


def plot_distances():
    all_data = get_data(
        Path(
            "/mnt/mass/max/BR09_CTdata/mnt/mass/scratch/br09_data/"
            "BR9_scan_list.csv"
        ),
        Path("/mnt/mass/max/BR09_CTdata/mnt/mass/scratch/br09_data/"),
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

        if name[:-3] == "BR9_09911":
            print(name, sorted(data)[::-1])

    names = []
    distances = []
    max_distance = 0

    for key, value in grouped_distances.items():
        names.append(key)
        distances.append(value)

        max_distance = (
            max(value) if max(value) > max_distance else max_distance
        )

    for i in range(0, len(distances), 10):
        start = i
        end = i + 10 if len(distances) > i + 10 else len(distances)
        plt.violinplot(distances[start:end], showmeans=True, widths=0.9)
        plt.xticks(
            range(1, (end - start) + 1), names[start:end], rotation="vertical"
        )
        plt.ylim(0, max_distance + 10)
        plt.savefig("plot_distances_between_grains_{}.svg".format(i / 10))
        plt.clf()

    return grouped_distances


if __name__ == "__main__":
    main()
