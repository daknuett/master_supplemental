from pathlib import Path
import numpy as np
from itertools import product

import sys
sys.path.append("../")
from load_data import load_from

sys.path.append("../../statistics")

from statistic.statistic import make_bins, jackknife_std, tauint

from paths import out_path_unmod as out_path, meta_unmod


def c11(X, s):
    return np.array([X[:, s] * X[:, s + t]
                     for t in range(len(X[0]))])

def c33(X, s):
    return np.array([X[:, s]**3 * X[:, s + t]**3
                     for t in range(len(X[0]))])

def hatf_H(cfg, t, a, omega):
    V = omega * cfg[:,t]**2
    T = (-(cfg[:, t-1] - cfg[:,t])**2 / a**2 * np.sqrt(2*np.pi/a)
         + np.sqrt(np.pi) / 2 * (a/2)**(-3/2))
    return V + np.sqrt(2*np.pi*a) * T / 4 / np.pi 


omega = meta_unmod["omega"]
Delta = meta_unmod["Delta"]
beta = meta_unmod["beta"]
n_tau = meta_unmod["n_tau"]
n_markov = meta_unmod["n_markov"]

delta_t = beta/n_tau

for i in range(100):
    data = load_from(out_path, str(i) + ".bindata", n_tau, n_markov)

        
    H_data = hatf_H(data, 0, delta_t, omega)
    tauint_H = tauint(H_data)
    del(H_data)

    tauint_q2 = tauint(data[:,0]**2)
    tauint_q4 = tauint(data[:,0]**4)

    c11_data = c11(data, 0)
    tauints_11 = [tauint(c) for c in c11_data]
    del(c11_data)

    c33_data = c33(data, 0)
    tauints_33 = [tauint(c) for c in c33_data]
    del(c33_data)

    with open("results_autocorrelation.tsv", "a") as fout:
        print(i, tauint_H, tauint_q2, tauint_q4, " ".join(str(t) for t in tauints_11), " ".join(str(t) for t in tauints_33), file=fout)

print("done")
