from pathlib import Path
import numpy as np

from data_list import n_taus, out_path, n_markov

import sys
sys.path.append("../")
from load_data import load_from

sys.path.append("../../statistics")
from statistic.statistic import make_bins, jackknife_std


omega = 0.5
beta = 10
Delta = 1
n_bins = 200

def S(Xi, omega2, delta_t, is_euclidean=False):
    if(len(Xi.shape) == 1):
        axis = 0
        idx = slice(None, -1, 1)
    else:
        axis = 1
        idx = slice(None, None, 1), slice(None, -1, 1)
    T = 0.5 * np.sum(((Xi - np.roll(Xi, -1, axis=axis))**2)[idx], axis=axis)
    #V = 0.5 * omega2 * np.sum((Xi**2 + np.roll(Xi, -1, axis=axis)**2)[idx], axis=axis)
    V = omega2 * np.sum((np.roll(Xi, -1, axis=axis)**2)[idx], axis=axis)

    if(is_euclidean):
        return T / delta_t + V * delta_t
    return T / delta_t - V * delta_t

def S_inlet(Xi, omega2, delta_t, ifrom, ito, is_euclidean=False):
    if(len(Xi.shape) == 1):
        return S(Xi[ifrom:ito], omega2, delta_t, is_euclidean=is_euclidean)
    return S(Xi[:, ifrom:ito], omega2, delta_t, is_euclidean=is_euclidean)


for n_tau in n_taus:
    delta_t = beta/n_tau
    data = load_from(out_path, str(n_tau) + ".bindata", n_tau, n_markov)

    E0_expect = 0.5
    Z_m_Z_mean = np.average((.5 - S_inlet(data, omega, delta_t, 0, 0+2)))
    Z_m_Z_std = np.std((.5 - S_inlet(data, omega, delta_t, 0, 0+2))) / np.sqrt(n_markov)
    E0_mean = Z_m_Z_mean / delta_t
    E0_std = Z_m_Z_std / delta_t
    E0_tension = (E0_mean - .5) / E0_std

    with open("results_H_lehner.csv", "a") as fout:
        print(n_tau, E0_mean, E0_std, beta, file=fout)
