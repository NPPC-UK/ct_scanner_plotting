from statistics import mean
import math

import numpy as np
from scipy import integrate


def _list_of_props(containers, fn):
    ls = []
    for container in containers:
        ls += fn(container)

    return ls


class Grain_Container:
    """Abstract base class that generalises between Pods, Treatments and
    Genotypes.

    All of these things contain a set of Grains, and all of them can
    calculate mean properties of the grains.
    """

    def mean_sphericity(self):

        return (
            mean(self.sphericities()) if self.n_grains() != 0 else float("nan")
        )

    def mean_surface_area(self):
        return (
            mean(self.surface_areas())
            if self.n_grains() != 0
            else float("nan")
        )

    def mean_volume(self):
        return mean(self.volumes()) if self.n_grains() != 0 else float("nan")

    def sphericities(self):
        raise NotImplementedError()

    def surface_areas(self):
        raise NotImplementedError()

    def volumes(self):
        raise NotImplementedError()

    def n_grains(self):
        return len(self.grains)


class Pod(Grain_Container):
    def __init__(self, grains, top, bottom, name):
        self.grains = []
        self.top = Point(*list(top))
        self.bottom = Point(*list(bottom))
        self.name = name
        self.spine = None

        for grain in grains:
            g_obj = Grain(grain)

            if (
                g_obj.position.z < self.bottom.z
                or g_obj.position.z > self.top.z
            ):
                raise ValueError(
                    "Grain {} is outside pod limits".format(g_obj)
                )

            self.grains.append(g_obj)

    @classmethod
    def pod_from_files(cls, grains_file, length_file, name):
        length = np.genfromtxt(length_file, delimiter=",", skip_header=0)
        return cls(
            np.genfromtxt(grains_file, delimiter=",", skip_header=1),
            length[4:],
            length[1:4],
            name,
        )

    def volumes(self):
        return [g.volume for g in self.grains]

    def surface_areas(self):
        return [g.surface_area for g in self.grains]

    def sphericities(self):
        return [g.sphericity() for g in self.grains]

    def filter(self):
        # It is not possible to fit the spine accurately after filtering, so do
        # now.
        if self.spine is None:
            self.fit()

        self.grains = [
            grain
            for grain, near_ends in zip(self.grains, self._near_ends())
            if not near_ends
        ]

    def n_grains(self):
        return len(self.grains)

    def length(self):
        return (self.top - self.bottom).norm()

    def _near_ends(self):
        near_ends = []
        for idx, grain in enumerate(self.grains):
            bottom_dist = (grain.position - self.bottom).norm()
            top_dist = (grain.position - self.top).norm()

            near_ends.append(
                bottom_dist < 0.02 * self.real_length()
                or top_dist < 0.02 * self.real_length()
            )

        return near_ends

    def __eq__(self, other):
        if len(self.grains) != len(other.grains):
            return False

        grains_equal = all(
            [
                s_grain == o_grain
                for s_grain, o_grain in zip(self.grains, other.grains)
            ]
        )

        return (
            grains_equal
            and self.top == other.top
            and self.bottom == other.bottom
        )

    def __str__(self):
        format_str = "{}, {}, {}, {}, {}, {}, {}, {}"
        return format_str.format(
            self.name,
            self.length(),
            self.n_grains(),
            self.mean_sphericity(),
            self.mean_volume(),
            self.mean_surface_area(),
            self.real_length(),
            self.n_grains() / self.real_length(),
        )

    def _arc_length_integrand(self, p):
        if self.spine is None:
            self.fit()

        return math.sqrt(
            self.spine[0].deriv()(p) ** 2 + self.spine[1].deriv()(p) ** 2 + 1
        )

    def real_zs(self):
        zs = [grain.position.z for grain in self.grains]

        return [
            integrate.quad(self._arc_length_integrand, self.bottom.z, z_cur)[0]
            for z_cur in zs
        ]

    def real_length(self):
        return integrate.quad(
            self._arc_length_integrand, self.bottom.z, self.top.z
        )[0]

    def fit(self):
        xs = [grain.position.x for grain in self.grains]
        xs.append(self.top.x)
        xs.insert(0, self.bottom.x)
        ys = [grain.position.y for grain in self.grains]
        ys.append(self.top.y)
        ys.insert(0, self.bottom.y)
        zs = [grain.position.z for grain in self.grains]
        zs.append(self.top.z)
        zs.insert(0, self.bottom.z)

        x_params = np.polyfit(zs, xs, 3)
        y_params = np.polyfit(zs, ys, 3)

        self.spine = (np.poly1d(x_params), np.poly1d(y_params))

    def scale(self, factor):
        """Scale all dimensions of the pod by a factor of 'factor'

        The behaviour of the pod is not defined if the spine has been
        previously fitted!
        """
        for grain in self.grains:
            grain.scale(factor)

        self.top.scale(factor)
        self.bottom.scale(factor)


class Plant(Grain_Container):
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

    def n_grains(self):
        n_grains = []
        for pod in self.pods:
            n_grains.append(pod.n_grains())

        return n_grains

    def mean_n_grains(self):
        return mean(self.n_grains())

    def real_zs(self):
        zs = []
        for pod in self.pods:
            zs += pod.real_zs()

        return zs


class Genotype(Grain_Container):
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

    def n_grains(self):
        return _list_of_props(self.plants, Plant.n_grains)

    def real_zs(self):
        zs = []
        for plant in self.plants:
            zs += plant.real_zs()

        return zs


class Grain:
    def __init__(self, grain):
        self.position = Point(*list(grain[9:12]))
        self.volume = grain[5]
        self.surface_area = grain[7]

    def sphericity(self):
        return (
            math.pi ** (1.0 / 3) * (6 * self.volume) ** (2.0 / 3)
        ) / self.surface_area

    def __eq__(self, other):
        return (
            self.position == other.position
            and self.volume == other.volume
            and self.surface_area == other.surface_area
        )

    def scale(self, factor):
        self.position.scale(factor)
        self.volume /= factor ** 3
        self.surface_area /= factor ** 2


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
