import numpy as np

from ct_plotting.data import Pod
import pytest

def setup_module(module):
    module.grain_data = np.array(
        [[4.81600000e-01, 1.32230554e+00, 1.20479616e+00, 3.64212345e-01,
          1.00935932e+00, 4.36385300e-01, 7.94433970e-02, 3.07358037e+00,
          0.00000000e+00, 1.62054478e+02, 3.61048507e+02, 1.52574627e+01,
          1.00000000e+00, 1.00000000e+00],
         [1.30720000e+00, 1.90259316e+00, 1.11549836e+00, 6.87062282e-01,
          7.65582535e-01, 1.48989757e+00, 2.93552273e-01, 8.28036437e+00,
          0.00000000e+00, 2.17175776e+02, 3.20451028e+02, 1.65248798e+02,
          1.00000000e+00, 1.00000000e+00],
         [1.03200000e+00, 1.26462054e+00, 8.55682359e-01, 8.16055067e-01,
          9.19376852e-01, 6.76722876e-01, 1.42822273e-01, 4.63877120e+00,
          0.00000000e+00, 2.25443215e+02, 3.05551492e+02, 2.30842637e+02,
          1.00000000e+00, 1.00000000e+00],
         [1.16960000e+00, 1.08909190e+00, 7.16141516e-01, 1.07392223e+00,
          9.98203035e-01, 8.03079217e-01, 7.94433970e-02, 4.60090368e+00,
          0.00000000e+00, 2.26165045e+02, 3.15074209e+02, 2.38196269e+02,
          1.00000000e+00, 1.00000000e+00],
         [1.51360000e+00, 1.86768546e+00, 1.67222963e+00, 8.10414831e-01,
          5.88994927e-01, 2.18811406e+00, 3.48488531e-01, 1.17326199e+01,
          0.00000000e+00, 2.25548013e+02, 3.10981688e+02, 2.63815989e+02,
          1.00000000e+00, 1.00000000e+00],
         [1.30720000e+00, 1.82025529e+00, 1.46079720e+00, 7.18141024e-01,
          9.56577861e-01, 1.84356506e+00, 7.94433970e-02, 8.24249685e+00,
          0.00000000e+00, 2.29444798e+02, 3.08318495e+02, 2.99194135e+02,
          1.00000000e+00, 1.00000000e+00],
         [1.44480000e+00, 1.65666879e+00, 1.51093672e+00, 8.72111558e-01,
          8.65680985e-01, 1.99629992e+00, 2.83726418e-01, 1.13350110e+01,
          0.00000000e+00, 2.44760144e+02, 2.98006217e+02, 3.24042212e+02,
          1.00000000e+00, 1.00000000e+00],
         [1.37600000e+00, 1.86968496e+00, 1.66717659e+00, 7.35952864e-01,
          8.09828577e-01, 1.86961792e+00, 2.67515096e-01, 1.06092169e+01,
          0.00000000e+00, 2.38063077e+02, 3.01297439e+02, 3.51384039e+02,
          1.00000000e+00, 1.00000000e+00],
         [1.30720000e+00, 1.96527300e+00, 1.38248744e+00, 6.65149319e-01,
          7.42622924e-01, 1.44821301e+00, 4.18123934e-01, 8.52650325e+00,
          0.00000000e+00, 2.58754836e+02, 2.91119883e+02, 3.73589519e+02,
          1.00000000e+00, 1.00000000e+00],
         [1.58240000e+00, 2.02794531e+00, 1.77237721e+00, 7.80297176e-01,
          4.32364543e-01, 2.31251643e+00, 4.53442522e-01, 1.19787588e+01,
          0.00000000e+00, 2.52073088e+02, 2.96984228e+02, 4.05052669e+02,
          1.00000000e+00, 1.00000000e+00],
         [1.65120000e+00, 2.02678481e+00, 1.69712667e+00, 8.14689349e-01,
          6.31872057e-01, 2.21937748e+00, 4.82475166e-01, 1.16884412e+01,
          3.25660672e-04, 2.52187702e+02, 2.91307896e+02, 4.41531846e+02,
          1.00000000e+00, 1.00000000e+00],
         [1.78880000e+00, 1.94740813e+00, 1.73456876e+00, 9.18554243e-01,
          9.62459978e-01, 2.90945244e+00, 7.94433970e-02, 1.14107460e+01,
          0.00000000e+00, 2.55942674e+02, 2.90525914e+02, 4.72785730e+02,
          1.00000000e+00, 1.00000000e+00],
         [1.85760000e+00, 1.98671109e+00, 1.14188966e+00, 9.35012649e-01,
          8.01492060e-01, 3.12308584e+00, 2.04612590e-01, 1.54247031e+01,
          0.00000000e+00, 2.70697393e+02, 2.85512930e+02, 5.14745777e+02,
          1.00000000e+00, 1.00000000e+00],
         [1.72000000e+00, 2.14684108e+00, 2.04215912e+00, 8.01177142e-01,
          6.97455976e-01, 3.17421457e+00, 4.48242844e-01, 1.54436369e+01,
          3.25660672e-04, 2.61594190e+02, 2.89014064e+02, 5.44977107e+02,
          1.00000000e+00, 1.00000000e+00],
         [1.37600000e+00, 2.11910078e+00, 1.83609605e+00, 6.49332026e-01,
          9.19018540e-01, 2.64273635e+00, 1.55910715e-01, 1.17831100e+01,
          0.00000000e+00, 2.77212832e+02, 2.87108698e+02, 5.72645404e+02,
          1.00000000e+00, 1.00000000e+00],
         [6.19200000e-01, 1.46684220e+00, 1.15539776e+00, 4.22131297e-01,
          9.58600391e-01, 5.91725441e-01, 7.94433970e-02, 5.98306816e+00,
          0.00000000e+00, 2.82297193e+02, 2.86756742e+02, 6.01293341e+02,
          1.00000000e+00, 1.00000000e+00],
         [1.03200000e+00, 1.80832340e+00, 1.28528371e+00, 5.70694379e-01,
          5.17489782e-01, 1.16065464e+00, 4.14532774e-01, 7.61137152e+00,
          3.25660672e-04, 2.81474319e+02, 2.85044625e+02, 6.11051361e+02,
          1.00000000e+00, 1.00000000e+00],
         [1.92640000e+00, 2.12825431e+00, 1.68968088e+00, 9.05154985e-01,
          6.76493541e-01, 3.11429301e+00, 3.97467540e-01, 1.53994581e+01,
          3.25660672e-04, 2.63203870e+02, 2.93568305e+02, 6.34140272e+02,
          1.00000000e+00, 1.00000000e+00],
         [1.78880000e+00, 1.74072654e+00, 1.65681375e+00, 1.02761689e+00,
          9.86722195e-01, 2.59779518e+00, 7.94433970e-02, 9.64359509e+00,
          0.00000000e+00, 2.65019935e+02, 2.94809178e+02, 7.07731444e+02,
          1.00000000e+00, 1.00000000e+00],
         [1.51360000e+00, 1.70871130e+00, 1.17746618e+00, 8.85813773e-01,
          8.42511991e-01, 1.66249773e+00, 2.19968806e-01, 9.27754240e+00,
          0.00000000e+00, 2.82666993e+02, 2.94632517e+02, 7.41592556e+02,
          1.00000000e+00, 1.00000000e+00],
         [1.65120000e+00, 2.00719847e+00, 1.54087484e+00, 8.22639127e-01,
          9.19086999e-01, 2.69647036e+00, 7.94433970e-02, 1.03125879e+01,
          0.00000000e+00, 2.65284575e+02, 2.98957000e+02, 7.68864356e+02,
          1.00000000e+00, 1.00000000e+00],
         [2.06400000e+00, 1.93795127e+00, 1.89294962e+00, 1.06504226e+00,
          7.82533187e-01, 3.32499546e+00, 4.69422578e-01, 1.34808371e+01,
          0.00000000e+00, 2.71378257e+02, 3.02264643e+02, 8.25868168e+02,
          1.00000000e+00, 1.00000000e+00],
         [1.78880000e+00, 1.97448590e+00, 1.82326140e+00, 9.05957345e-01,
          9.52457660e-01, 3.33704491e+00, 1.07226677e-01, 1.25593941e+01,
          0.00000000e+00, 2.82044859e+02, 3.02546618e+02, 8.64349394e+02,
          1.00000000e+00, 1.00000000e+00],
         [2.06400000e-01, 1.70975512e+00, 1.55129458e+00, 1.20719042e-01,
          9.47861890e-01, 2.36103987e-01, 1.31490580e-01, 3.40176555e+00,
          0.00000000e+00, 2.81068966e+02, 3.05953103e+02, 9.24528276e+02,
          1.00000000e+00, 1.00000000e+00],
         [1.85760000e+00, 1.14378518e+00, 1.09326094e+00, 1.62408119e+00,
          1.04384500e+00, 1.48468700e+00, 7.94433970e-02, 9.29016491e+00,
          0.00000000e+00, 2.82914016e+02, 3.02092564e+02, 9.50587629e+02,
          1.00000000e+00, 1.00000000e+00]] 
    )
    module.length = np.array([2000, 0, 0, 0, 0, 0, 2000])
    module.pod = Pod(module.grain_data, module.length[4:], length[1:4], 
                     'TestPod')


def test_calculate_correct_volume():
    assert (pytest.approx(pod.mean_volume(), 1e-8) == 
            np.mean(grain_data, axis=0)[5])

def test_calculate_correct_surface_area():
    assert (pytest.approx(pod.mean_surface_area(), 1e-8) == 
            np.mean(grain_data, axis=0)[7])

def test_sphericities_returns_type_list():
    assert type([]) == type(pod.sphericities())

def test_volumes_returned_in_insertion_order():
    assert list(pod.volumes()) == list(grain_data[:, 5])

def test_pod_can_load_grain_from_file(tmpdir):
    np.savetxt(tmpdir / 'grain.csv', grain_data, delimiter=',',
               header='some, header, dont, worry, about, it')
    np.savetxt(tmpdir / 'lengths.csv', length, delimiter=',')

    p_file = Pod.pod_from_files(Pod, tmpdir / 'grain.csv', 
                                tmpdir / 'lengths.csv', 
                                'TestName')

    p_direct = Pod(grain_data, length[4:], length[1:4], 'TestName')
    assert p_direct == p_file


def test_pod_rejects_impossible_pod_geometry():
    with pytest.raises(ValueError):
        pod = Pod(grain_data, [0, 0, 1], [0, 0, 0], 'ImpossiblePod')


def test_pod_filters_grains_less_than_10_from_ends():
    implausible_grains = np.append(
        grain_data,
        [
            [4.81600000e-01, 1.32230554e+00, 1.20479616e+00, 3.64212345e-01,
             1.00935932e+00, 4.36385300e-01, 7.94433970e-02, 3.07358037e+00,
             0.00000000e+00, length[1], length[2], length[3]+9.9999999,
             1.00000000e+00, 1.00000000e+00],
            [4.81600000e-01, 1.32230554e+00, 1.20479616e+00, 3.64212345e-01,
             1.00935932e+00, 4.36385300e-01, 7.94433970e-02, 3.07358037e+00,
             0.00000000e+00, length[4], length[5], length[6]-9.9999999,
             1.00000000e+00, 1.00000000e+00]
        ],
        axis=0
    )

    print(implausible_grains)
    imp_pod = Pod(implausible_grains, length[4:], length[1:4], 'Implausible')

    filtered_pod = imp_pod.filter()
    assert filtered_pod == pod
