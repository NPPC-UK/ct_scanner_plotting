from math import sqrt

import numpy as np
from scipy import integrate
import seaborn as sns
import matplotlib.pyplot as plt


def exp_fit_positions(pods):
    real_lengths = []
    real_zs = []

    for pod in pods:
        x_pos = [grain.position.x for grain in pod.grains]
        x_pos.append(pod.top.x)
        x_pos.insert(0, pod.bottom.x)
        y_pos = [grain.position.y for grain in pod.grains]
        y_pos.append(pod.top.y)
        y_pos.insert(0, pod.bottom.y)
        z_pos = [grain.position.z for grain in pod.grains]
        z_pos.append(pod.top.z)
        z_pos.insert(0, pod.bottom.z)

        x_params = np.polyfit(z_pos, x_pos, 3)
        y_params = np.polyfit(z_pos, y_pos, 3)

        x = np.poly1d(x_params)
        y = np.poly1d(y_params)

        def arc_length_integrand(p):
            return sqrt((x.deriv()(p) ** 2 + y.deriv()(p) ** 2 + 1))

        real_length, length_error = integrate.quad(
            arc_length_integrand, min(z_pos), max(z_pos)
        )
        real_lengths.append(real_length)

        real_zs.append(
            [
                integrate.quad(arc_length_integrand, min(z_pos), z_cur)[0]
                for z_cur in z_pos[1:-1]
            ]
        )

    sns.set_style("whitegrid")
    sns.swarmplot(data=real_zs, size=0.5)
    plt.savefig("./dist.svg")
    plt.clf()
