from pathlib import Path
import numpy as np

import os
import sqlite3

from scipy.linalg import eigh as seigh, eig as seig

import sys
sys.path.append("../")
from load_data import load_from

sys.path.append("../../statistics")

from statistic.statistic import make_bins, jackknife_std, tauint
from statistic.fitting import fit

from paths import out_path_mod_gs as out_path_mod, meta_mod_gs as meta_mod

omega = meta_mod["omega"]
Delta = meta_mod["Delta"]
beta = meta_mod["beta"]
n_tau = meta_mod["n_tau"]
n_markov = meta_mod["n_markov"]
tinsert = meta_mod["tinsert"]
Eb = meta_mod["Eb"]


delta_t = beta / n_tau

n_bins = 2000
binsize = n_markov / n_bins

def eigh(a, b=None, **kwargs):
    try:
        return seigh(a, b=b, **kwargs)
    except:
        abar = np.linalg.inv(b) @ a
        return seigh(abar, **kwargs)


def epsf_H(X, tinsert, omega2, delta_t):
    return 0.5 - (
        (X[:, tinsert + 1] - X[:, tinsert])**2 / 2 / delta_t
        - (X[:, tinsert + 1]**2 + X[:, tinsert]**2) / 2 * omega2 * delta_t
    )    
    

def c11(X, s):
    return (np.array([np.sign(epsf_H(X, tinsert, omega, delta_t) - Eb*delta_t) * X[:, s]**3
                            * X[:, s + t]**3
                     for t in range(len(X[0]))]) 
            / np.average(np.sign(epsf_H(X, tinsert, omega, delta_t) - Eb*delta_t)))
def c12(X, s):
    return (np.array([np.sign(epsf_H(X, tinsert, omega, delta_t) - Eb*delta_t) * X[:, s]**3
                            * X[:, s + t]**5
                     for t in range(len(X[0]))])
            / np.average(np.sign(epsf_H(X, tinsert, omega, delta_t) - Eb*delta_t)))
def c21(X, s):
    return (np.array([np.sign(epsf_H(X, tinsert, omega, delta_t) - Eb*delta_t) * X[:, s]**5
                            * X[:, s + t]**3
                     for t in range(len(X[0]))])
            / np.average(np.sign(epsf_H(X, tinsert, omega, delta_t) - Eb*delta_t)))
def c22(X, s):
    return (np.array([np.sign(epsf_H(X, tinsert, omega, delta_t) - Eb*delta_t) * X[:, s]**5
                            * X[:, s + t]**5
                     for t in range(len(X[0]))])
            / np.average(np.sign(epsf_H(X, tinsert, omega, delta_t) - Eb*delta_t)))

def C(X, s):
    return np.array([[c11(X, s), c12(X, s)]
                     , [c21(X, s), c22(X, s)]])


def effective_mass_decay(C, eps, i):
    X = C[i] / C[i + 1]
    effective_mass = np.log(X) / eps
   
    return effective_mass

def GEVP_corrf(means, T, delta):
    data = np.array([eigh(means[:, :, i], b=means[:, :, i - delta], eigvals_only=True) for i in range(delta, means.shape[2] // 2)])
    
    gs = data[:, 1]
    es = data[:, 0]
    
    return np.array([gs, es])

def GEVP_effective_masses(means, T, delta):
    data1 = np.array([eigh(means[:, :, i], b=means[:, :, i - delta], eigvals_only=True) for i in range(delta, means.shape[2] // 2)])
    data2 = np.array([eigh(means[:, :, i + 1], b=means[:, :, i - delta], eigvals_only=True) for i in range(delta, means.shape[2] // 2)])


    eps = T[1] - T[0]
    
    gs_me = -np.log(data2[:, 1] / data1[:, 1]) / eps
    es_me = -np.log(data2[:, 0] / data1[:, 0]) / eps
    
    return np.array([gs_me[:-1], es_me[:-1]])


i = os.environ["SLURM_ARRAY_TASK_ID"]

ensemble = load_from(out_path_mod, str(i) + ".bindata", n_tau, n_markov)
corrf_data = C(ensemble, 0)
corrf_data = make_bins(corrf_data, n_bins)
T = np.arange(0, beta, delta_t)

means = np.mean(corrf_data, axis=3)

delta = 1
data = GEVP_effective_masses(means, T, delta)
stds = jackknife_std(corrf_data, GEVP_effective_masses, T, delta, data_axis=3, eps=1e-4)

non_nan_es = np.where(~ np.isnan(stds[1,:]))
is_nan_es = np.where(np.isnan(stds[1,:]))
non_nan_gs = np.where(~ np.isnan(stds[0,:]))
is_nan_gs = np.where(np.isnan(stds[0,:]))

y = lambda t, p: np.ones_like(t) * p[0]
p0 = np.array([2])
n_skip_gs = 2
ntake_gs = n_skip_gs + 2

n_skip_es = 2
ntake_es = n_skip_es + 2

p_gs, p_std_gs, f_std_gs, return_data_gs = fit(y
                                                , T[delta:means.shape[2]//2 - 1][non_nan_gs][n_skip_gs:ntake_gs]
                                                , data[0,:][non_nan_gs][n_skip_gs:ntake_gs]
                                                , stds[0,:][non_nan_gs][n_skip_gs:ntake_gs]
                                                , p0
                                                , statistic=lambda means: GEVP_effective_masses(means, T, delta)[0,:][non_nan_gs][n_skip_gs:ntake_gs]
                                                , data=corrf_data
                                                , fit_kwargs={"eps1": 1e-6, "eps2": 1e-6, "eps3": 1e-5, "eps4": 1e-5, "nmax": 150}
                                                , jackknife_kwargs={"data_axis": 3, "eps": 1e-4})

p_es, p_std_es, f_std_es, return_data_es = fit(y
                                                , T[delta:means.shape[2]//2 - 1][non_nan_es][n_skip_es:ntake_es]
                                                , data[1,:][non_nan_es][n_skip_es:ntake_es]
                                                , stds[1,:][non_nan_es][n_skip_es:ntake_es]
                                                , p0
                                                , statistic=lambda means: GEVP_effective_masses(means, T, delta)[1,:][non_nan_es][n_skip_es:ntake_es]
                                                , data=corrf_data
                                                , fit_kwargs={"eps1": 1e-6, "eps2": 1e-6, "eps3": 1e-5, "eps4": 1e-5, "nmax": 150}
                                                , jackknife_kwargs={"data_axis": 3, "eps": 1e-4}
                                                )

db = sqlite3.connect("result.sqlite3")

# method TEXT, result_name TEXT, ensemble_id INT, ensemble_collection TEXT, result NUM, std NUM
cur = db.execute(" INSERT INTO results VALUES(?, ?, ?, ?, ?, ?)", ("extract_GEVP_q3q5_delta_mod.py", "gs", i, str(out_path_mod), p_gs[0], p_std_gs[0]))
cur.execute(" INSERT INTO results VALUES(?, ?, ?, ?, ?, ?)", ("extract_GEVP_q3q5_delta_mod.py", "es", i, str(out_path_mod), p_es[0], p_std_es[0]))
db.commit()
db.close()
