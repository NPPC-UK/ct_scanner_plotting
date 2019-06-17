from statistics import mean
import math

import numpy as np
from scipy import integrate
import jenks_natural_breaks


def _list_of_props(containers, fn):
    ls = []
    for container in containers:
        ls += fn(container)

    return ls


class Seed_Container:
    """Abstract base class that generalises between Pods, Treatments and
    Genotypes.

    All of these things contain a set of Seeds, and all of them can
    calculate mean properties of the seeds.
    """

    def mean_sphericity(self):

        return (
            mean(self.sphericities()) if self.n_seeds() != 0 else float("nan")
        )

    def mean_surface_area(self):
        return (
            mean(self.surface_areas()) if self.n_seeds() != 0 else float("nan")
        )

    def mean_volume(self):
        return mean(self.volumes()) if self.n_seeds() != 0 else float("nan")

    def sphericities(self):
        raise NotImplementedError()

    def surface_areas(self):
        raise NotImplementedError()

    def volumes(self):
        raise NotImplementedError()

    def n_seeds(self):
        return len(self.seeds)


class Pod(Seed_Container):
    def __init__(self, seeds, dims, name):
        self.seeds = []
        self.slices = [Slice(*s) for s in dims]
        self.dims = np.asarray(dims)
        self.name = name
        self.spine = None
        self._real_length = None

        for seed in seeds:
            g_obj = Seed(seed)

            if (
                g_obj.position.z < self._bottom().z
                or g_obj.position.z > self._top().z
            ):
                raise ValueError("Seed {} is outside pod limits".format(g_obj))

            self.seeds.append(g_obj)

    @classmethod
    def pod_from_files(cls, seeds_file, length_file, name):
        dims = np.genfromtxt(length_file, delimiter=",", skip_header=1)
        return cls(
            np.genfromtxt(seeds_file, delimiter=",", skip_header=1), dims, name
        )

    def _top(self):
        return self.slices[-1].position

    def _bottom(self):
        return self.slices[0].position

    def volumes(self):
        return [g.volume for g in self.seeds]

    def surface_areas(self):
        return [g.surface_area for g in self.seeds]

    def sphericities(self):
        return [g.sphericity() for g in self.seeds]

    def filter(self):
        good_seeds = []
        for seed in self.seeds:
            near_slice_idx = np.searchsorted(self.dims[:, 2], seed.position.z)
            if abs(self.dims[near_slice_idx, 2] - seed.position.z) > 10:
                good_seeds.append(seed)
            elif self.dims[near_slice_idx, 4] > 3 * seed.radius():
                good_seeds.append(seed)

        self.seeds = good_seeds

        breaks = jenks_natural_breaks.classify(np.array(self.real_zs()), 0)
        print(breaks)
        self.seeds = [
            seed for seed in self.seeds if self._real_z(seed) > breaks[0]
        ]

    def n_seeds(self):
        return len(self.seeds)

    def length(self):
        return (self._top() - self._bottom()).norm()

    def _near_ends(self):
        near_ends = []
        for idx, seed in enumerate(self.seeds):
            bottom_dist = (seed.position - self._bottom()).norm()
            top_dist = (seed.position - self._top()).norm()

            near_ends.append(
                bottom_dist < 0.02 * self.real_length()
                or top_dist < 0.02 * self.real_length()
            )

        return near_ends

    def __eq__(self, other):
        if len(self.seeds) != len(other.seeds):
            return False

        seeds_equal = all(
            [
                s_seed == o_seed
                for s_seed, o_seed in zip(self.seeds, other.seeds)
            ]
        )

        return (
            seeds_equal
            and self._top() == other._top()
            and self._bottom() == other._bottom()
        )

    def __str__(self):
        format_str = "{}, {}, {}, {}, {}, {}, {}, {}"
        return format_str.format(
            self.name,
            self.length(),
            self.n_seeds(),
            self.mean_sphericity(),
            self.mean_volume(),
            self.mean_surface_area(),
            self.real_length(),
            self.n_seeds() / self.real_length(),
        )

    def _arc_length_integrand(self, p):
        if self.spine is None:
            self.fit()

        return math.sqrt(
            self.spine[0].deriv()(p) ** 2 + self.spine[1].deriv()(p) ** 2 + 1
        )

    def _real_z(self, seed):
        if seed.real_z is None:
            seed.real_z = integrate.quad(
                self._arc_length_integrand, self._bottom().z, seed.position.z
            )[0]

        return seed.real_z

    def real_zs(self):
        return [self._real_z(seed) for seed in self.seeds]

    def real_length(self):
        if self._real_length is None:
            self._real_length = integrate.quad(
                self._arc_length_integrand, self._bottom().z, self._top().z
            )[0]

        return self._real_length

    def fit(self):
        xs = [s.position.x for s in self.slices]
        ys = [s.position.y for s in self.slices]
        zs = [s.position.z for s in self.slices]

        x_params = np.polyfit(zs, xs, 3)
        y_params = np.polyfit(zs, ys, 3)

        self.spine = (np.poly1d(x_params), np.poly1d(y_params))

    def scale(self, factor):
        """Scale all dimensions of the pod by a factor of 'factor'

        The behaviour of the pod is not defined if the spine has been
        previously fitted!
        """
        for seed in self.seeds:
            seed.scale(factor)

        self._top().scale(factor)
        self._bottom().scale(factor)

    def _sort_seeds(self):
        self.seeds.sort(key=lambda seed: self._real_z(seed))

    def seed_spacings(self):
        spacings = []
        for seed_1, seed_2 in zip(self.seeds[:-1], self.seeds[1:]):
            spacings.append(self._real_z(seed_2) - self._real_z(seed_1))

        return spacings


class Plant(Seed_Container):
    @classmethod
    def group_from_pods(cls, pods, name_fn):
        plants = []
        grouped = {}

        for pod in pods:
            name = name_fn(pod.name)
            if name not in grouped:
                grouped[name] = [pod]
            else:
                grouped[name].append(pod)

        for name, pod_group in grouped.items():
            plants.append(cls(pod_group, name))

        return plants

    def __init__(self, pods, name):
        self.pods = pods
        self.name = name

    def volumes(self):
        return _list_of_props(self.pods, Pod.volumes)

    def sphericities(self):
        return _list_of_props(self.pods, Pod.sphericities)

    def surface_areas(self):
        return _list_of_props(self.pods, Pod.surface_areas)

    def n_seeds(self):
        n_seeds = []
        for pod in self.pods:
            n_seeds.append(pod.n_seeds())

        return n_seeds

    def mean_n_seeds(self):
        return mean(self.n_seeds())

    def real_zs(self):
        zs = []
        for pod in self.pods:
            zs += pod.real_zs()

        return zs

    def seed_spacings(self):
        vs = []
        for pod in self.pods:
            vs += pod.seed_spacings()

        return vs


class Genotype(Seed_Container):
    @classmethod
    def group_from_plants(cls, plants, name_fn):
        genotypes = []
        grouped = {}

        for plant in plants:
            name = name_fn(plant.name)
            if name not in grouped:
                grouped[name] = [plant]
            else:
                grouped[name].append(plant)

        for name, plant_group in grouped.items():
            genotypes.append(cls(plant_group, name))

        return genotypes

    def __init__(self, plants, name):
        self.plants = plants
        self.name = name

    def sphericities(self):
        return _list_of_props(self.plants, Plant.sphericities)

    def surface_areas(self):
        return _list_of_props(self.plants, Plant.surface_areas)

    def volumes(self):
        return _list_of_props(self.plants, Plant.volumes)

    def n_seeds(self):
        return _list_of_props(self.plants, Plant.n_seeds)

    def real_zs(self):
        zs = []
        for plant in self.plants:
            zs += plant.real_zs()

        return zs

    def seed_spacings(self):
        vs = []
        for plant in self.plants:
            vs += plant.seed_spacings()

        return vs


class Seed:
    def __init__(self, seed):
        self.position = Point(*list(seed[9:12]))
        self.volume = seed[5]
        self.surface_area = seed[7]
        self.real_z = None

    def sphericity(self):
        return (
            math.pi ** (1.0 / 3) * (6 * self.volume) ** (2.0 / 3)
        ) / self.surface_area

    def __eq__(self, other):
        print(self.position, other.position)
        return (
            self.position == other.position
            and self.volume == other.volume
            and self.surface_area == other.surface_area
        )

    def scale(self, factor):
        self.position.scale(factor)
        self.volume /= factor ** 3
        self.surface_area /= factor ** 2

    def radius(self):
        return (3.0 / 4.0) * (1.0 / math.pi) * self.volume ** (1.0 / 3.0)


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return type(self)(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return type(self)(self.x - other.x, self.y - other.y, self.z - other.z)

    def scale(self, factor):
        self.x /= factor
        self.y /= factor
        self.z /= factor

    def __repr__(self):
        return "Point({}, {}, {})".format(self.x, self.y, self.z)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def norm(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)


class Slice:
    def __init__(self, x, y, z, minor, major):
        self.position = Point(x, y, z)
        self.major = major
        self.minor = minor
