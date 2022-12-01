from pathlib import Path
import numpy as np
from itertools import product

from data_list import n_taus, out_path, n_markov, beta

import sys
sys.path.append("../")
from load_data import load_from

sys.path.append("../../statistics")

from statistic.statistic import make_bins, jackknife_std, tauint, jackknife2_std


def hatf_H(cfg, t, a, omega):
    V = omega * cfg[:,t]**2
    T = (-(cfg[:, t-1] - cfg[:,t])**2 / a**2 * np.sqrt(2*np.pi/a)
         + np.sqrt(np.pi) / 2 * (a/2)**(-3/2))
    return V + np.sqrt(2*np.pi*a) * T / 4 / np.pi 


omega = 0.5
beta = 10 
n_bins = 8000
binsize = n_markov / n_bins / 4  # We only use every 2nd configuration.


for n_tau in n_taus:
    delta_t = beta/n_tau
    data = load_from(out_path, str(n_tau) + ".bindata", n_tau, n_markov)
    data = data[::4,:]
    print(tauint(data[:,0]))


    H_data = hatf_H(data, 0, delta_t, omega)
    H_data = make_bins(H_data, n_bins)
    variance_H = np.var(H_data) * binsize
    variance_H_err = jackknife2_std(H_data, lambda x: np.var(x)*binsize , data_axis=0)
    del(H_data)

    data_q = data[:,0]
    variance_q = np.var(data_q) 
    data_q = make_bins(data_q, n_bins)
    #variance_q = np.var(data_q) * binsize
    variance_q_err = jackknife2_std(data_q, lambda x: np.var(x)*binsize , data_axis=0)

    data_q2 = data[:,0]**2
    #variance_q2 = np.var(data_q2) 
    data_q2 = make_bins(data_q2, n_bins)
    variance_q2 = np.var(data_q2) * binsize
    variance_q2_err = jackknife2_std(data_q2, lambda x: np.var(x)*binsize , data_axis=0)

    data_qqp = data[:,0] - data[:,1]
    data_qqp = make_bins(data_qqp, n_bins)
    variance_qqp = np.var(data_qqp) * binsize
    variance_qqp_err = jackknife2_std(data_qqp, lambda x: np.var(x)*binsize , data_axis=0)

    data_qqp2 = (data[:,0] - data[:,1])**2
    data_qqp2 = make_bins(data_qqp2, n_bins)
    variance_qqp2 = np.var(data_qqp2) * binsize
    variance_qqp2_err = jackknife2_std(data_qqp2, lambda x: np.var(x)*binsize , data_axis=0)

    with open("results_fluctuations.tsv", "a") as fout:
        print(n_tau, delta_t, variance_H, variance_H_err, variance_q, variance_q_err, variance_q2, variance_q2_err, variance_qqp, variance_qqp_err, variance_qqp2, variance_qqp2_err, file=fout)

print("done")
