from statistics import mean
import math

import numpy as np


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
        return mean(self.sphericities())

    def mean_surface_area(self):
        return mean(self.surface_areas())

    def mean_volume(self):
        return mean(self.volumes())

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

            near_ends.append(bottom_dist < 10 or top_dist < 10)

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
        format_str = "{}, {}, {}, {}, {}, {}"
        return format_str.format(
            self.name,
            (self.top - self.bottom).norm(),
            self.n_grains(),
            self.mean_sphericity(),
            self.mean_volume(),
            self.mean_surface_area(),
        )


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


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return type(self)(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return type(self)(self.x - other.x, self.y - other.y, self.z - other.z)

    def __repr__(self):
        return "Point({}, {}, {})".format(self.x, self.y, self.z)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def norm(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
