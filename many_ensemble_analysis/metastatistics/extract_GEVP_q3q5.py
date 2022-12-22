from pathlib import Path
import numpy as np
from itertools import product

from scipy.linalg import eigh as seigh, eig as seig

import sys
sys.path.append("../")
from load_data import load_from

sys.path.append("../../statistics")

from statistic.statistic import make_bins, jackknife_std, tauint
from statistic.fitting import fit

from paths import out_path_unmod, meta_unmod

omega = meta_unmod["omega"]
Delta = meta_unmod["Delta"]
beta = meta_unmod["beta"]
n_tau = meta_unmod["n_tau"]
n_markov = meta_unmod["n_markov"]

delta_t = beta / n_tau

n_bins = 2000
binsize = n_markov / n_bins



def eigh(a, b=None, **kwargs):
    try:
        return seigh(a, b=b, **kwargs)
    except:
        abar = np.linalg.inv(b) @ a
        return seigh(abar, **kwargs)


def c11(X, s):
    return np.array([X[:, s]**3
                            * X[:, s + t]**3
                     for t in range(len(X[0]))]) 
def c12(X, s):
    return np.array([X[:, s]**3
                            * X[:, s + t]**5
                     for t in range(len(X[0]))]) 
def c21(X, s):
    return np.array([X[:, s]**5
                            * X[:, s + t]**3
                     for t in range(len(X[0]))]) 
def c22(X, s):
    return np.array([X[:, s]**5
                            * X[:, s + t]**5
                     for t in range(len(X[0]))])

def C(X, s):
    return np.array([[c11(X, s), c12(X, s)]
                     , [c21(X, s), c22(X, s)]])


def effective_mass_decay(C, eps, i):
    X = C[i] / C[i + 1]
    effective_mass = np.log(X) / eps
   
    return effective_mass

def GEVP_corrf(means, T, n0):
    C0 = means[:, :, n0]
    data = np.array([eigh(means[:, :, i], b=C0, eigvals_only=True) for i in range(n0 + 1, means.shape[2]//2)])
    
    gs = data[:, 1]
    es = data[:, 0]
    
    return np.array([gs, es])

def GEVP_effective_masses(means, T, n0):
    C0 = means[:, :, n0]
    data = np.array([eigh(means[:, :, i], b=C0, eigvals_only=True) for i in range(1, means.shape[2]//2)])
    
    gs = data[:, 1]
    es = data[:, 0]
    eps = T[1] - T[0]
    
    gs_me = np.array([effective_mass_decay(gs, eps, i) for i in range(T.shape[0]//2 - 2)])
    es_me = np.array([effective_mass_decay(es, eps, i) for i in range(T.shape[0]//2 - 2)])
    
    return np.array([gs_me, es_me])


results_gs = []
results_es = []


for i in range(100):
    ensemble = load_from(out_path_unmod, str(i) + ".bindata", n_tau, n_markov)
    corrf_data = C(ensemble, 0)
    corrf_data = make_bins(corrf_data, n_bins)
    T = np.arange(0, beta, delta_t)

    means = np.mean(corrf_data, axis=3)

    n0 = 2
    data = GEVP_effective_masses(means, T, n0)
    stds = jackknife_std(corrf_data, GEVP_effective_masses, T, n0, data_axis=3)

    non_nan_es = np.where(~ np.isnan(stds[1,:]))
    is_nan_es = np.where(np.isnan(stds[1,:]))
    non_nan_gs = np.where(~ np.isnan(stds[0,:]))
    is_nan_gs = np.where(np.isnan(stds[0,:]))

    y = lambda t, p: np.ones_like(t) * p[0]
    p0 = np.array([2])
    n_skip = 3

    p_gs, p_std_gs, f_std_gs, return_data_gs = fit(y
                    , T[1:means.shape[2]//2 - 1][non_nan_gs][n_skip:]
                    , data[0,:][non_nan_gs][n_skip:]
                    , stds[0,:][non_nan_gs][n_skip:]
                    , p0
                    , statistic=lambda means: GEVP_effective_masses(means, T, n0)[0,:][non_nan_gs][n_skip:]
                    , data=corrf_data
                    , jackknife_kwargs={"data_axis": 3})

    p_es, p_std_es, f_std_es, return_data_es = fit(y
                    , T[1:means.shape[2]//2 - 1][non_nan_es][n_skip:]
                    , data[1,:][non_nan_es][n_skip:]
                    , stds[1,:][non_nan_es][n_skip:]
                    , p0
                    , statistic=lambda means: GEVP_effective_masses(means, T, n0)[1,:][non_nan_es][n_skip:]
                    , data=corrf_data
                    , jackknife_kwargs={"data_axis": 3}
                    )

    results_gs.append([p_gs[0], p_std_gs[0]])
    results_es.append([p_es[0], p_std_es[0]])


results_gs = np.array(results_gs)
results_es = np.array(results_es)


with open("results_GEVP_q3q5.csv", "w") as fout:
    print("gs", results_gs.shape[0], np.mean(results_gs[:,0]), np.std(results_gs[:,0]), np.mean(results_gs[:,1]), np.std(results_gs[:,1]), file=fout)
    print("es", results_es.shape[0], np.mean(results_es[:,0]), np.std(results_es[:,0]), np.mean(results_es[:,1]), np.std(results_es[:,1]), file=fout)

results = np.array([results_gs, results_es])
np.save("results_GEVP_q3q5.npy", results)
