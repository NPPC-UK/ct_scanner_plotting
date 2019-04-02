import numpy as np

class Pod:

    grains = []
    top = []
    bottom = []
    name = ""

    def __init__(self, grains, top, bottom, name):
        self.grains = []
        self.top = top
        self.bottom = bottom 
        self.name = name

        for grain in grains:
            self.grains.append(Grain(grain))

        self.top = top
        self.bottom = bottom

    def pod_from_files(cls, grains_file, length_file, name):
        length = np.genfromtxt(length_file, delimitier=',', skip_header=0),
        return cls(
            np.genfromtxt(grains_file, delimitier=',', skip_header=1),
            name
        )

    def mean_volume():
        return mean(self.volumes())

    def mean_surface_area():
        return mean(self.surface_areas())

    def mean_sphericity():
        return mean(self.sphericities())

    def volumes(self):
        return [g.volume for g in self.grains]

    def surface_areas(self):
        return [g.surface_area for g in self.grains]

    def sphericities(self):
        return [g.sphericity() for g in self.grains]

class Grain:
    position = []
    volume = 0
    surface_area = 0

    def __init__(self, grain):
        self.position = grain[9:12]
        self.volume = grain[5]
        self.surface_area = [7]

    def sphericity():
        return (math.pi**(1./3) * (6*self.volume)**(2./3))/self.surface_area
