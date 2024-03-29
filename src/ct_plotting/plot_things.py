from pathlib import Path
import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from ct_plotting.plots import (
    plot_sorted_property,
    plot_property_vs_property,
    plot_pearson_correlations,
    plot_bar_property,
    plot_swarm_property,
    plot_spine_debug,
    plot_kde_debug,
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
    23: "Boxplot of silique length grouped by genotype",
    24: "Boxplot of beak length grouped by genotype",
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


def get_data(meta_file, base_path, plant_name_fn):
    meta_data = np.genfromtxt(
        meta_file,
        delimiter="\t",
        usecols=[0, 4, 6],
        dtype=str,
        skip_header=1,
        comments="#disabled#",
    )

    pods = []
    genotype_lookup = {}

    print(meta_data[0])
    for scan in meta_data:
        genotype_lookup[plant_name_fn(scan[0])] = scan[2]
        csv_dir = base_path / scan[1]

        # Glob returns a generator.  I know that there is only one file
        # matching.  I collect on the entire generator and retrieve the first
        # item.
        try:
            seeds_file = list(csv_dir.glob("*.ISQ*-raw_stats.csv"))[0]
        except IndexError:
            print(f"Couldn't find seeds file {csv_dir}")
            continue

        try:
            dims_file = list(csv_dir.glob("*.ISQ*dims.csv"))[0]
        except IndexError:
            print(f"Couldn't find dims file {csv_dir}")
            continue

        pod = Pod.pod_from_files(seeds_file, dims_file, scan[0])

        pods.append(pod)

    return pods, genotype_lookup


def plot(pods, plants, genotypes, outdir, plot, genotype_lookup, scale=1.0):
    def save(fig, fname):
        fig.savefig(outdir / "plot_{}.svg".format(fname))
        fig.clf()

    if plot == 0:
        save(
            plot_bar_property(
                genotypes,
                lambda x: [v / scale ** 3 for v in Genotype.volumes(x)],
                property_name="volumes of seeds",
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
                lambda x: [s / scale ** 2 for s in Genotype.surface_areas(x)],
                property_name="surface area of seeds",
            ),
            "bar_surface_area_of_seeds",
        )
    elif plot == 4:
        save(
            plot_sorted_property(
                pods,
                lambda x: Pod.mean_volume(x) / scale ** 3,
                property_name="mean volume of seeds",
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
                lambda x: Plant.mean_volume(x) / scale ** 3,
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
                pods,
                lambda x: Pod.length(x) / scale,
                property_name="length of pod",
            ),
            "length_of_pod",
        )
    elif plot == 10:
        save(
            plot_property_vs_property(
                pods,
                lambda x: Pod.length(x) / scale,
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
                lambda x: Pod.length(x) / scale,
                lambda x: Pod.mean_volume(x) / scale ** 3,
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
                lambda x: Pod.mean_volume(x) / scale ** 3,
                "number of seeds",
                "mean volume of seeds",
            ),
            "number_of_seeds_vs_mean_volume_of_seeds",
        )
    elif plot == 13:
        save(
            plot_property_vs_property(
                pods,
                lambda x: Pod.mean_volume(x) / scale ** 3,
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
                lambda x: Pod.mean_surface_area(x) / scale ** 2,
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
                    Pod.silique_length,
                    Pod.beak_length,
                    Pod.n_seeds,
                    Pod.mean_volume,
                    Pod.mean_sphericity,
                    Pod.mean_surface_area,
                    lambda pod: pod.n_seeds() / pod.silique_length(),
                ],
                [
                    "widths",
                    "length",
                    "silique_length",
                    "beak_length",
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
            plot_swarm_property(
                plants,
                lambda x: [z / scale for z in Plant.real_zs(x)],
                "seed position",
            ),
            "real_zs_plant",
        )
    elif plot == 17:
        save(
            plot_swarm_property(
                genotypes,
                lambda x: [z / scale for z in Genotype.real_zs(x)],
                "seed position",
            ),
            "real_zs_genotype",
        )
    elif plot == 18:

        def densities(genotype):
            ds = []
            for plant in genotype.plants:
                for pod in plant.pods:
                    ds.append(pod.n_seeds() / (pod.silique_length() / scale))

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
                    vs.append(sum(pod.volumes()) / scale ** 3)

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
                    vs.append(pod.real_length() / scale)
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
                lambda x: [s / scale for s in Genotype.seed_spacings(x)],
                property_name="seed spacings",
            ),
            "bar_seed_spacings",
        )
    elif plot == 22:
        save(
            plot_bar_property(
                genotypes,
                lambda x: [w / scale for w in Genotype.pod_widths(x)],
                property_name="pod widths",
            ),
            "bar_pod_widths",
        )
    elif plot == 23:

        def pod_lengths(genotype):
            vs = []
            for plant in genotype.plants:
                for pod in plant.pods:
                    vs.append(pod.silique_length() / scale)
            return vs

        save(
            plot_bar_property(
                genotypes, pod_lengths, property_name="silique length"
            ),
            "bar_silique_lengths",
        )
    elif plot == 24:

        def pod_lengths(genotype):
            vs = []
            for plant in genotype.plants:
                for pod in plant.pods:
                    vs.append(pod.beak_length() / scale)
            return vs

        save(
            plot_bar_property(
                genotypes, pod_lengths, property_name="beak length"
            ),
            "bar_beak_lengths",
        )


def run(args):
    def plant_name(name):
        if args.pod_suffix_length == 0:
            return name
        else:
            return name[: -args.pod_suffix_length]

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

    pods, genotype_lookup = get_data(meta_file, args.working_dir, plant_name)

    plants = Plant.group_from_pods(pods, plant_name)
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
        plot(
            pods,
            plants,
            genotypes,
            args.output_dir,
            p,
            genotype_lookup,
            args.scale,
        )

    for p in pods:
        if p.name in args.plot_spine_debug:
            plot_spine_debug(p, "spine_debug_{}".format(p.name))

    for g in genotypes:
        if g.name in args.plot_kde_debug:
            plot_kde_debug(g, "kde_debug_{}".format(g.name))

    if args.print_stats:
        print(
            "Name, Length, N_Seeds, Sphericity, Volume, Surface Area, "
            "Real Length, Silique Length, Beak Length, Width, "
            "Density (N_Seeds/Silique Length), Genotype"
        )
        for pod in pods:
            print(str(pod), ",", f'"{genotype_lookup[plant_name(pod.name)]}"')

        if args.extended_stats:
            for pod in pods:
                with open(
                    args.output_dir / f"{pod.name}_spacings.csv", "w"
                ) as csvfile:
                    csv.writer(csvfile).writerows(
                        [[spacing] for spacing in pod.seed_spacings()]
                    )
                with open(
                    args.output_dir / f"{pod.name}_seed_realz.csv", "w"
                ) as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Real Z", "X", "Y", "Z"])
                    writer.writerows(
                        [
                            [
                                seed.real_z,
                                seed.position.x,
                                seed.position.y,
                                seed.position.z,
                            ]
                            for seed in pod.seeds
                        ]
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
        "--plot_spine_debug",
        action="append",
        nargs="+",
        default=[[]],
        type=str,
        metavar="Pod",
        help="plot spine fitting debug graphs for the given Pods",
    )

    parser.add_argument(
        "--plot_kde_debug",
        action="append",
        nargs="+",
        default=[[]],
        type=str,
        metavar="Genotype",
        help="plot kde graphs for the given Genotypes",
    )

    parser.add_argument(
        "-P",
        "--plot",
        action="append",
        nargs="+",
        default=[],
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
        "--extended_stats",
        action="store_true",
        help="print seed level stats about the brassica pods",
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
    parser.add_argument(
        "--pod_suffix_length",
        action="store",
        type=int,
        default=2,
        help=(
            "The number of characters at the end of the bar code that "
            "identify the pod.  If there is only one pod per plant, pass 0"
        ),
    )

    args = parser.parse_args()
    args.plot = set([item for sublist in args.plot for item in sublist])
    args.plot_spine_debug = set(
        [item for sublist in args.plot_spine_debug for item in sublist]
    )
    args.plot_kde_debug = set(
        [item for sublist in args.plot_kde_debug for item in sublist]
    )

    return args


def main():
    args = get_arguments()
    run(args)


if __name__ == "__main__":
    main()
