from statistics import mean
import math

import numpy as np


class Pod:
    def __init__(self, grains, top, bottom, name):
        self.grains = []
        self.top = list(top)
        self.bottom = list(bottom)
        self.name = name

        for grain in grains:
            self.grains.append(Grain(grain))

    def pod_from_files(cls, grains_file, length_file, name):
        length = np.genfromtxt(length_file, delimiter=',', skip_header=0)
        return cls(
            np.genfromtxt(grains_file, delimiter=',', skip_header=1),
            length[1:4],
            length[4:],
            name
        )

    def mean_volume(self):
        return mean(self.volumes())

    def mean_surface_area(self):
        return mean(self.surface_areas())

    def mean_sphericity(self):
        return mean(self.sphericities())

    def volumes(self):
        return [g.volume for g in self.grains]

    def surface_areas(self):
        return [g.surface_area for g in self.grains]

    def sphericities(self):
        return [g.sphericity() for g in self.grains]

    def __eq__(self, other):
        grains_equal = all(
            [s_grain == o_grain for s_grain, o_grain in 
             zip(self.grains, other.grains)]
        )

        return (grains_equal and 
                self.top == other.top and 
                self.bottom == other.bottom)


class Grain:
    def __init__(self, grain):
        self.position = grain[9:12]
        self.volume = grain[5]
        self.surface_area = grain[7]

    def sphericity(self):
        return (math.pi**(1./3) * (6*self.volume)**(2./3))/self.surface_area

    def __eq__(self, other):
        return (all(self.position == other.position) and
                self.volume == other.volume and
                self.surface_area == other.surface_area)
