from pathlib import Path
import numpy as np

from data_list import n_taus, out_path, n_markov, beta

import sys
sys.path.append("../")
from load_data import load_from

sys.path.append("../../statistics")
from statistic.statistic import make_bins, jackknife_std

def hatf_O(cfg, t, a, omega):
    V = omega * cfg[:,t]**2
    T = (-(cfg[:, t-1] - cfg[:,t])**2 / a**2 * np.sqrt(2*np.pi/a)
         + np.sqrt(np.pi) / 2 * (a/2)**(-3/2))
    return V + np.sqrt(2*np.pi*a) * T / 4 / np.pi 

omega = 0.5
Delta = 1
n_bins = 200



for n_tau in n_taus:
    delta_t = beta/n_tau
    data = load_from(out_path, str(n_tau) + ".bindata", n_tau, n_markov)

        
    H_data = hatf_O(data, 0, delta_t, omega)
    H_data = make_bins(H_data, n_bins)
    H_mean = np.mean(H_data)
    H_std = np.std(H_data) / np.sqrt(H_data.shape[0])


    with open("results_H_lehner.csv", "a") as fout:
        print(n_tau, H_mean, H_std, beta, file=fout)
