from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from ct_plotting.plots import (
    plot_sorted_property,
    plot_property_vs_property,
    plot_pearson_correlations,
    plot_bar_property,
    plot_swarm_property,
)

from ct_plotting.data import Pod, Plant, Genotype


available_plots = {
    0: "Boxplot of grain volumes grouped by genotype",
    1: "Boxplot of number of grains grouped by genotype",
    2: "Boxplot of sphericities grouped by genotype",
    3: "Boxplot of surface areas grouped by genotype",
    4: "Sorted pod mean of grain volume",
    5: "Sorted number of grains in pods",
    6: "Sorted plant mean of grain volume",
    7: "Sorted number of grains in plants",
    8: "Sorted pod mean of grain sphericities",
    9: "Sorted pod length",
    10: "Number of grains against pod length",
    11: "Pod length against pod mean volume of grains",
    12: "Number of grains against pod mean volume of grains",
    13: "Pod mean volume of grains against pod mean sphericity of grains",
    14: "Pod mean surface area of grains against pod mean sphericities of "
    "grains",
    15: "Correlation matrix of all the calculable grain properties",
    16: "Grain positions grouped by plant",
    17: "Grain positions grouped by genotype",
    18: "Boxplot of grain density grouped by genotype",
    19: "Boxplot of total grain volume in pods grouped by genotype",
}


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
        [
            ("sample_name", np.unicode_, 12),
            ("folder", np.unicode_, 8),
            ("genotype", np.unicode_, 50),
        ]
    )

    meta_data = np.genfromtxt(
        meta_file,
        delimiter="\t",
        usecols=[0, 4, 6],
        dtype=meta_type,
        skip_header=1,
    )

    pods = []
    genotype_lookup = {}

    for scan in meta_data:
        genotype_lookup[scan[0][:-2]] = scan[2]
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

        pod = Pod.pod_from_files(grains_file, length_file, scan[0])

        pods.append(pod)

    return pods, genotype_lookup


def plot(pods, outdir, plot, genotype_lookup):
    plants = Plant.group_from_pods(pods, lambda name: name[:-2])
    genotypes = Genotype.group_from_plants(
        plants, lambda name: genotype_lookup[name]
    )

    def save(fig, fname):
        fig.savefig(outdir / "plot_{}.svg".format(fname))
        fig.clf()

    if plot == 0:
        save(
            plot_bar_property(
                genotypes, Genotype.volumes, property_name="volumes of grains"
            ),
            "bar_volumes_of_grains",
        )
    elif plot == 1:
        save(
            plot_bar_property(
                genotypes, Genotype.n_grains, property_name="number of grains"
            ),
            "bar_number_of_grains",
        )
    elif plot == 2:
        save(
            plot_bar_property(
                genotypes,
                Genotype.sphericities,
                property_name="sphericities of grains",
            ),
            "bar_sphericities_of_grains",
        )
    elif plot == 3:
        save(
            plot_bar_property(
                genotypes,
                Genotype.surface_areas,
                property_name="surface area of grains",
            ),
            "bar_surface_area_of_grains",
        )
    elif plot == 4:
        save(
            plot_sorted_property(
                pods, Pod.mean_volume, property_name="mean volume of grains"
            ),
            "mean_volume_of_grains",
        )
    elif plot == 5:
        save(
            plot_sorted_property(
                pods, Pod.n_grains, property_name="number of grains"
            ),
            "number_of_grains",
        )
    elif plot == 6:
        save(
            plot_sorted_property(
                plants,
                Plant.mean_volume,
                property_name="grouped mean volume of grains",
            ),
            "grouped_mean_volume_of_grains",
        )
    elif plot == 7:
        save(
            plot_sorted_property(
                plants,
                Plant.mean_n_grains,
                property_name="grouped number of grains",
            ),
            "group_number_of_grains",
        )
    elif plot == 8:
        save(
            plot_sorted_property(
                pods,
                Pod.mean_sphericity,
                property_name="mean sphericity of grains",
            ),
            "mean_sphericity_of_grains",
        )
    elif plot == 9:
        save(
            plot_sorted_property(
                pods, Pod.length, property_name="length of pod"
            ),
            "length_of_pod",
        )
    elif plot == 10:
        save(
            plot_property_vs_property(
                pods,
                Pod.length,
                Pod.n_grains,
                "length of pod",
                "number of grains",
            ),
            "length_of_pod_vs_number_of_grains",
        )
    elif plot == 11:
        save(
            plot_property_vs_property(
                pods,
                Pod.length,
                Pod.mean_volume,
                "length of pod",
                "mean volume of grains",
            ),
            "length_of_pod_vs_mean_volume_of_grains",
        )
    elif plot == 12:
        save(
            plot_property_vs_property(
                pods,
                Pod.n_grains,
                Pod.mean_volume,
                "number of grains",
                "mean volume of grains",
            ),
            "number_of_grains_vs_mean_volume_of_grains",
        )
    elif plot == 13:
        save(
            plot_property_vs_property(
                pods,
                Pod.mean_volume,
                Pod.mean_sphericity,
                "mean volume of grains",
                "mean sphericities",
            ),
            "mean_volume_of_grains_vs_mean_sphericities",
        )
    elif plot == 14:
        save(
            plot_property_vs_property(
                pods,
                Pod.mean_surface_area,
                Pod.mean_sphericity,
                "mean surface area of grains",
                "mean sphericities",
            ),
            "mean_surface_area_of_grains_vs_mean_sphericities",
        )
    elif plot == 15:
        save(
            plot_pearson_correlations(
                pods,
                [
                    Pod.length,
                    Pod.n_grains,
                    Pod.mean_volume,
                    Pod.mean_sphericity,
                    Pod.mean_surface_area,
                    lambda pod: pod.n_grains() / pod.length(),
                ],
                [
                    "length",
                    "n_grains",
                    "volumes",
                    "sphericities",
                    "surface_areas",
                    "grain_density",
                ],
            ),
            "correlations",
        )
    elif plot == 16:
        save(
            plot_swarm_property(plants, Plant.real_zs, "grain position"),
            "real_zs_plant",
        )
    elif plot == 17:
        save(
            plot_swarm_property(genotypes, Genotype.real_zs, "grain position"),
            "real_zs_genotype",
        )
    elif plot == 18:

        def densities(genotype):
            ds = []
            for plant in genotype.plants:
                for pod in plant.pods:
                    ds.append(pod.n_grains() / pod.length())

            return ds

        save(
            plot_bar_property(
                genotypes, densities, property_name="densities of grains"
            ),
            "bar_density_of_grains",
        )
    elif plot == 19:

        def sum_grain_volumes(genotype):
            vs = []
            for plant in genotype.plants:
                for pod in plant.pods:
                    vs.append(sum(pod.volumes()))

            return vs

        save(
            plot_bar_property(
                genotypes,
                sum_grain_volumes,
                property_name="sum of grain volume per pod",
            ),
            "bar_sum_grain_volumes_in_pod",
        )


def main(args):
    if args.list:
        print("Possible plots:")
        for key, description in available_plots.items():
            print("{}:\t{}".format(key, description))

        return

    meta_file = (
        args.meta_file
        if args.meta_file.is_absolute()
        else args.working_dir / args.meta_file
    )

    pods, genotype_lookup = get_data(meta_file, args.working_dir)

    if args.filter:
        for pod in pods:
            pod.filter()

    for p in args.plot:
        plot(pods, args.output_dir, p, genotype_lookup)

    if args.print_stats:
        print(
            "Name, Length, N_Grains, Sphericity, Volume, Surface Area, "
            "Real Length, Density (N_Grains/Real Length), Genotype"
        )
        for pod in pods:
            print(str(pod), ",", genotype_lookup[pod.name[:-2]])


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


def get_arguments():
    parser = argparse.ArgumentParser(
        description="A program to plot stats of microCT scans of Brassica pods"
    )

    parser.add_argument(
        "-l", "--list", action="store_true", help="list the possible plots"
    )

    parser.add_argument(
        "-P",
        "--plot",
        action="append",
        nargs="+",
        default=[[]],
        choices=available_plots,
        type=int,
        help="select which plots to plot",
    )

    parser.add_argument(
        "-p",
        "--print_stats",
        action="store_true",
        help="print stats about the brassica pods",
    )
    parser.add_argument(
        "-d",
        "--working_dir",
        type=Path,
        default=Path("./"),
        metavar="DIR",
        help="path to the working directory, uses the current "
        "directory by default",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=Path("./"),
        metavar="DIR",
        help="path to the output directory, uses the current "
        "directory by default.  This is where the plots are "
        "saved too.",
    )
    parser.add_argument(
        "-m",
        "--meta_file",
        type=Path,
        default="meta.csv",
        metavar="FILE",
        help="path to the file containing the meta data, relative "
        "paths start from the WORKING_DIR",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="remove implausibly positioned grains",
    )

    args = parser.parse_args()
    args.plot = set([item for sublist in args.plot for item in sublist])

    return args


if __name__ == "__main__":
    args = get_arguments()
    main(args)
