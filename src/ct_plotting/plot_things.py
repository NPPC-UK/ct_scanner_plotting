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
    0: "Boxplot of seed volumes grouped by genotype",
    1: "Boxplot of number of seeds grouped by genotype",
    2: "Boxplot of sphericities grouped by genotype",
    3: "Boxplot of surface areas grouped by genotype",
    4: "Sorted pod mean of seed volume",
    5: "Sorted number of seeds in pods",
    6: "Sorted plant mean of seed volume",
    7: "Sorted number of seeds in plants",
    8: "Sorted pod mean of seed sphericities",
    9: "Sorted pod length",
    10: "Number of seeds against pod length",
    11: "Pod length against pod mean volume of seeds",
    12: "Number of seeds against pod mean volume of seeds",
    13: "Pod mean volume of seeds against pod mean sphericity of seeds",
    14: "Pod mean surface area of seeds against pod mean sphericities of "
    "seeds",
    15: "Correlation matrix of all the calculable seed properties",
    16: "Seed positions grouped by plant",
    17: "Seed positions grouped by genotype",
    18: "Boxplot of seed density grouped by genotype",
    19: "Boxplot of total seed volume in pods grouped by genotype",
    20: "Boxplot of pod length grouped by genotype",
    21: "Boxplot of seed spacings grouped by genotype",
    22: "Boxplot of pod widths grouped by genotype",
}


def merge_seeds(seeds):
    """Return a seed corresponding to the 'sum' of an arbitrary number of
    other seeds.

    Many parameters are meaningless of a merged seed.  Only the center of
    mass, and total volume make complete sense.  Surface area is more iffy.
    Should we count the surface of the seed fragments that are facing each
    other? The simple sum is calculated here.

    Return a merged seeds that has null values except where explained above.
    """
    xc, yc, zc = 0, 0, 0

    xc = sum([seed[5] * seed[9] for seed in seeds]) / sum(
        [seed[5] for seed in seeds]
    )

    yc = sum([seed[5] * seed[10] for seed in seeds]) / sum(
        [seed[5] for seed in seeds]
    )

    zc = sum([seed[5] * seed[11] for seed in seeds]) / sum(
        [seed[5] for seed in seeds]
    )
    return np.array(
        [
            -1,
            -1,
            -1,
            -1,
            -1,
            sum([seed[5] for seed in seeds]),
            -1,
            sum([seed[7] for seed in seeds]),
            -1,
            xc,
            yc,
            zc,
            1,
            1,
        ]
    )


def filter_seeds(seeds, bottom, top):
    """Filter seeds of a pod.

    Delete those seeds that are too close to the ends of a pod to be real.
    Merge those whose centres are too close together to be real.

    Return the rest.
    """
    false_seeds_idxs = []

    for idx, seed in enumerate(seeds):
        bottom_dist = np.linalg.norm(seed[9:12] - bottom)
        top_dist = np.linalg.norm(seed[9:12] - top)

        near_ends = bottom_dist < 100 or top_dist < 100
        if near_ends:
            false_seeds_idxs.append(idx)

    filtered_seeds = np.delete(seeds, false_seeds_idxs, 0)

    seed_merge_candidates = []

    for idx, seed in enumerate(filtered_seeds):
        distances = np.linalg.norm(
            filtered_seeds[:, 9:12] - seed[9:12], axis=1
        )
        for dist_idx, dist in enumerate(distances):
            if dist_idx <= idx:
                continue

            if dist < 10:
                seed_merge_candidates.append((idx, dist_idx))

    G = nx.Graph()
    G.add_edges_from(seed_merge_candidates)
    to_delete = []
    for to_merge in nx.connected_components(G):
        filtered_seeds += merge_seeds([filtered_seeds[i] for i in to_merge])
        to_delete += to_merge

    return np.delete(filtered_seeds, to_delete, 0)


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
            seeds_file = list(csv_dir.glob("*.ISQ-raw.csv"))[0]
        except IndexError:
            continue

        try:
            dims_file = list(csv_dir.glob("*.ISQdims.csv"))[0]
        except IndexError:
            continue

        pod = Pod.pod_from_files(seeds_file, dims_file, scan[0])

        pods.append(pod)

    return pods, genotype_lookup


def plot(pods, plants, genotypes, outdir, plot, genotype_lookup):
    def save(fig, fname):
        fig.savefig(outdir / "plot_{}.svg".format(fname))
        fig.clf()

    if plot == 0:
        save(
            plot_bar_property(
                genotypes, Genotype.volumes, property_name="volumes of seeds"
            ),
            "bar_volumes_of_seeds",
        )
    elif plot == 1:
        save(
            plot_bar_property(
                genotypes, Genotype.n_seeds, property_name="number of seeds"
            ),
            "bar_number_of_seeds",
        )
    elif plot == 2:
        save(
            plot_bar_property(
                genotypes,
                Genotype.sphericities,
                property_name="sphericities of seeds",
            ),
            "bar_sphericities_of_seeds",
        )
    elif plot == 3:
        save(
            plot_bar_property(
                genotypes,
                Genotype.surface_areas,
                property_name="surface area of seeds",
            ),
            "bar_surface_area_of_seeds",
        )
    elif plot == 4:
        save(
            plot_sorted_property(
                pods, Pod.mean_volume, property_name="mean volume of seeds"
            ),
            "mean_volume_of_seeds",
        )
    elif plot == 5:
        save(
            plot_sorted_property(
                pods, Pod.n_seeds, property_name="number of seeds"
            ),
            "number_of_seeds",
        )
    elif plot == 6:
        save(
            plot_sorted_property(
                plants,
                Plant.mean_volume,
                property_name="grouped mean volume of seeds",
            ),
            "grouped_mean_volume_of_seeds",
        )
    elif plot == 7:
        save(
            plot_sorted_property(
                plants,
                Plant.mean_n_seeds,
                property_name="grouped number of seeds",
            ),
            "group_number_of_seeds",
        )
    elif plot == 8:
        save(
            plot_sorted_property(
                pods,
                Pod.mean_sphericity,
                property_name="mean sphericity of seeds",
            ),
            "mean_sphericity_of_seeds",
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
                Pod.n_seeds,
                "length of pod",
                "number of seeds",
            ),
            "length_of_pod_vs_number_of_seeds",
        )
    elif plot == 11:
        save(
            plot_property_vs_property(
                pods,
                Pod.length,
                Pod.mean_volume,
                "length of pod",
                "mean volume of seeds",
            ),
            "length_of_pod_vs_mean_volume_of_seeds",
        )
    elif plot == 12:
        save(
            plot_property_vs_property(
                pods,
                Pod.n_seeds,
                Pod.mean_volume,
                "number of seeds",
                "mean volume of seeds",
            ),
            "number_of_seeds_vs_mean_volume_of_seeds",
        )
    elif plot == 13:
        save(
            plot_property_vs_property(
                pods,
                Pod.mean_volume,
                Pod.mean_sphericity,
                "mean volume of seeds",
                "mean sphericities",
            ),
            "mean_volume_of_seeds_vs_mean_sphericities",
        )
    elif plot == 14:
        save(
            plot_property_vs_property(
                pods,
                Pod.mean_surface_area,
                Pod.mean_sphericity,
                "mean surface area of seeds",
                "mean sphericities",
            ),
            "mean_surface_area_of_seeds_vs_mean_sphericities",
        )
    elif plot == 15:
        save(
            plot_pearson_correlations(
                pods,
                [
                    Pod.width,
                    Pod.length,
                    Pod.n_seeds,
                    Pod.mean_volume,
                    Pod.mean_sphericity,
                    Pod.mean_surface_area,
                    lambda pod: pod.n_seeds() / pod.length(),
                ],
                [
                    "widths",
                    "length",
                    "n_seeds",
                    "volumes",
                    "sphericities",
                    "surface_areas",
                    "seed_density",
                ],
            ),
            "correlations",
        )
    elif plot == 16:
        save(
            plot_swarm_property(plants, Plant.real_zs, "seed position"),
            "real_zs_plant",
        )
    elif plot == 17:
        save(
            plot_swarm_property(genotypes, Genotype.real_zs, "seed position"),
            "real_zs_genotype",
        )
    elif plot == 18:

        def densities(genotype):
            ds = []
            for plant in genotype.plants:
                for pod in plant.pods:
                    ds.append(pod.n_seeds() / pod.silique_length())

            return ds

        save(
            plot_bar_property(
                genotypes, densities, property_name="densities of seeds"
            ),
            "bar_density_of_seeds",
        )
    elif plot == 19:

        def sum_seed_volumes(genotype):
            vs = []
            for plant in genotype.plants:
                for pod in plant.pods:
                    vs.append(sum(pod.volumes()))

            return vs

        save(
            plot_bar_property(
                genotypes,
                sum_seed_volumes,
                property_name="sum of seed volume per pod",
            ),
            "bar_sum_seed_volumes_in_pod",
        )
    elif plot == 20:

        def pod_lengths(genotype):
            vs = []
            for plant in genotype.plants:
                for pod in plant.pods:
                    vs.append(pod.real_length())
            return vs

        save(
            plot_bar_property(
                genotypes, pod_lengths, property_name="pod length"
            ),
            "bar_pod_lengths",
        )
    elif plot == 21:
        save(
            plot_bar_property(
                genotypes,
                Genotype.seed_spacings,
                property_name="seed spacings",
            ),
            "bar_seed_spacings",
        )
    elif plot == 22:
        save(
            plot_bar_property(
                genotypes, Genotype.pod_widths, property_name="pod widths"
            ),
            "bar_pod_widths",
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

    if args.scale != 1.0:
        for pod in pods:
            pod.scale(args.scale)

    plants = Plant.group_from_pods(pods, lambda name: name[:-2])
    genotypes = Genotype.group_from_plants(
        plants, lambda name: genotype_lookup[name]
    )

    if args.filter:
        for pod in pods:
            pod.filter()
        for plant in plants:
            plant.filter()
        for genotype in genotypes:
            genotype.filter()

    for p in args.plot:
        plot(pods, plants, genotypes, args.output_dir, p, genotype_lookup)

    if args.print_stats:
        print(
            "Name, Length, N_Seeds, Sphericity, Volume, Surface Area, "
            "Real Length, Density (N_Seeds/Real Length), Genotype"
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
            # Ignore very small separations.  These are not separate seeds
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
        plt.savefig("plot_distances_between_seeds_{}.svg".format(i / 10))
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
        help="remove implausibly positioned seeds",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=float,
        help="scale all distances and positions by dividing by this number",
        default=1.0,
    )
    parser.add_argument(
        "-t",
        "--from_top",
        action="store_true",
        default=True,
        help="all distances are measured from the top of the scan",
    )

    args = parser.parse_args()
    args.plot = set([item for sublist in args.plot for item in sublist])

    return args


if __name__ == "__main__":
    args = get_arguments()
    main(args)
