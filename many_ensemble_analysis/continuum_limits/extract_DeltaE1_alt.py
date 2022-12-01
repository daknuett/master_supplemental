from pathlib import Path
import numpy as np
from itertools import product

from data_list import n_taus, out_path, n_markov, beta

import sys
sys.path.append("../")
from load_data import load_from

sys.path.append("../../statistics")
from statistic.statistic import make_bins, jackknife_std
from statistic.fitting import fit

def c11(X, s):
    return np.array([X[:, s] * X[:, s + t]
                     for t in range(len(X[0]))])
def deltaE_decay(C, eps, i):
    X = C[i] / C[i + 1]
    deltaE = np.log(X) / eps
   
    return deltaE

omega = 0.5
Delta = 1
n_bins = 200
binsize = n_markov / n_bins

for n_tau in n_taus:
    delta_t = beta/n_tau
    data = load_from(out_path, str(n_tau) + ".bindata", n_tau, n_markov)

    corrf_data = c11(data, 0)
    corrf_data = make_bins(corrf_data, n_bins)
    means = np.mean(corrf_data, axis=1)
    var = np.var(corrf_data, axis=1) * binsize

    T = np.arange(0, beta, delta_t)    




    f = lambda t, p: p[0] * (np.exp(-p[1] * t) + np.exp(-p[1] * (beta - t)))
    p0 = np.array([1.1, 1.1]) 

    p, p_std, f_std, return_data = fit(f, T, means, np.sqrt(var / n_markov)
                                    , p0
                                    , statistic=lambda means: means
                                    , data=corrf_data
                                    )

    optif = return_data["f_optimal"](T)
    optie = return_data["std_estimator"](T)

    with open(f"plotlogs/cosh_fit_{n_tau}.py", "w") as fout:
        print("import matplotlib.pyplot as plt", file=fout)
        print("import numpy as np", file=fout)
        print("T = np.array(", list(T), ")", file=fout)
        print("V = np.array(", list(means), ")", file=fout)
        print("E = np.array(", list(np.sqrt(var / n_markov)), ")", file=fout)
        print("p = np.array(", list(optif), ")", file=fout)
        print("p_std = np.array(", list(optie), ")", file=fout)
        print("plt.errorbar(T, V, E, fmt='.')", file=fout)
        print("plt.fill_between(T, p+p_std, p-p_std, alpha=.5)", file=fout)
        print("plt.show()", file=fout)

    with open("results_DeltaE1_alt.tsv", "a") as fout:
        print(n_tau, p[1], p_std[1], beta, file=fout)
