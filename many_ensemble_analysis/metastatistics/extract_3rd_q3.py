from pathlib import Path
import numpy as np
from itertools import product

import sys
sys.path.append("../")
from load_data import load_from

sys.path.append("../../statistics")

from statistic.statistic import make_bins, jackknife_std, tauint
from statistic.fitting import fit

from paths import out_path_unmod, out_path_mod_gs as out_path_mod, meta_mod_gs as meta_mod

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


def c33(X, s):
    return np.array([X[:, s]**3
                            * X[:, s + t]**3
                     for t in range(len(X[0]))])

def deltaE_decay(C, eps, i):
    X = C[i] / C[i + 1]
    deltaE = np.log(X) / eps
   
    return deltaE

def epsf_H(X, tinsert, omega2, delta_t):
    return 0.5 - (
        (X[:, tinsert + 1] - X[:, tinsert])**2 / 2 / delta_t
        - (X[:, tinsert + 1]**2 + X[:, tinsert]**2) / 2 * omega2 * delta_t
    )

def c33_m(X, s, omega, delta_t, ifrom, Ebias):
    ito = ifrom + 2
    return (np.array([np.sign(epsf_H(X, ifrom, omega, delta_t) - Ebias)
                            * X[:, s]**3
                            * X[:, s + t]**3
                     for t in range(len(X[0]))])
        / np.average(np.sign(epsf_H(X, ifrom, omega, delta_t) - Ebias)))

def c33_hm(X, s, omega, delta_t, ifrom, Ebias):
    ito = ifrom + 2
    return (np.array([(epsf_H(X, ifrom, omega, delta_t) - Ebias)
                            * X[:, s]**3
                            * X[:, s + t]**3
                     for t in range(len(X[0]))])
        / np.average((epsf_H(X, ifrom, omega, delta_t) - Ebias)))

def effective_mass_decay(C, eps, i):
    X = C[i] / C[i + 1]
    effective_mass = np.log(X) / eps
   
    return effective_mass

results_um = []
results_hm = []
results_fm = []
for i in range(100):
    ensemble1 = load_from(out_path_unmod, str(i) + ".bindata", n_tau, n_markov)
    ensemble2 = load_from(out_path_mod, str(i) + ".bindata", n_tau, n_markov)

    corrf_data = c33(ensemble1, 0)
    corrf_data = make_bins(corrf_data, n_bins)
    
    correlation_f_mod1 = c33_hm(ensemble1, 0, omega, delta_t, tinsert, Eb*delta_t)
    correlation_f_mod1 = make_bins(correlation_f_mod1, n_bins)

    corrf = c33_m(ensemble2, 0, omega, delta_t, tinsert, Eb*delta_t)
    corrf = make_bins(corrf, n_bins)

    means = np.mean(corrf_data, axis=1)
    var = np.var(corrf_data, axis=1) * binsize

    T = np.arange(0, beta, delta_t)    


    decaying_regime = corrf_data[:corrf_data.shape[0] // 2]
    decaying_regime_mean = means[:corrf_data.shape[0] // 2]

    T_decaying = T[:corrf_data.shape[0] // 2]
    deltaEs = np.array([deltaE_decay(decaying_regime_mean, beta/n_tau, i) for i in range(decaying_regime_mean.size - 1)])
    deltaEs_std = np.array([jackknife_std(decaying_regime, lambda x: deltaE_decay(np.abs(x), beta/n_tau, i)) for i in range(decaying_regime_mean.size - 1)])

    decaying_std =  np.sqrt(var  / n_markov)[:corrf_data.shape[0] // 2]


    p0_guesses = np.array(list(product([0.4, 1, 2], [1.1, .1, 2], [0.2, 1, 3])))


    f = lambda t, p: p[0] + p[1]*np.exp(-t * (p[2] - p[0]))
    p0 = np.array([1, 2., 2.])
    nskip = 2
    nskipend = 2 + 4
    p, p_std, f_std, return_data = fit(f, T_decaying[nskip:-3 - nskipend], deltaEs[nskip:-2 - nskipend], deltaEs_std[nskip:-2 - nskipend], p0
                                , statistic=lambda means:np.array([deltaE_decay(means, beta/n_tau, i) for i in range(means.size - 1)])[nskip:-2 - nskipend]
                                , data=decaying_regime
                                , fit_kwargs={"eps1": 1e-5, "eps2": 1e-5, "eps3": 1e-3, "eps4": 1e-3}
                                , jk_use_central_result=True
                                )

    with open("3rd_um.log.csv", "a") as fout:
        print(p[2], p_std[2], file=fout, sep=",")
    results_um.append([p[2], p_std[2]])

    cf_mod = correlation_f_mod1
    correlation_f_mod1 = correlation_f_mod1[:T.shape[0] // 2]

    T_mod1 = T[:T.shape[0] // 2]
    corr_mod1_mean = np.mean(correlation_f_mod1, axis=1)

    def me(means, eps):
        return np.log(np.abs(means / np.roll(means, -1)))[:-1] / eps

    T_consider = T_mod1[1:]
    corrf_consider = corr_mod1_mean[1:]

    me_means = me(corrf_consider, T_mod1[1] - T_mod1[0])
    me_stds = jackknife_std(correlation_f_mod1[1:], me, T_mod1[1] - T_mod1[0], data_axis=1)

    nskipendplateau = 10

    y = lambda t, p: p[0] * np.ones_like(t)
    p_m, p_std_m, f_std_m, return_data_m = fit(y, T_consider[:-1 - nskipendplateau], me_means[:-nskipendplateau], me_stds[:-nskipendplateau]
                                            , np.array([2.])
                                            , statistic=lambda means: me(means, T_mod1[1] - T_mod1[0])[:-nskipendplateau]
                                            , data=correlation_f_mod1[1:]
                                            , jk_use_central_result=True
                                            , fit_kwargs={"eps1": 1e-5, "eps2": 1e-5, "eps3": 1e-3, "eps4": 1e-3}
                                            , error_estimating_kwargs={"collect_jk_samples": False}
                                            )

    results_hm.append([p_m[0], p_std_m[0]])

    T = np.arange(0, beta, delta_t)

    considered_corrf = corrf[tinsert + 1: T.shape[0] // 2]
    considered_T = T[tinsert + 1: T.shape[0] // 2]

    means = np.mean(considered_corrf, axis=1)
    dEdec = lambda cf: np.array([effective_mass_decay(cf, beta/n_tau, i) 
                         for i in range(considered_T.shape[0] - 1)])

    deltaEs_m = dEdec(means)

    deltaEs_std_m = jackknife_std(considered_corrf, dEdec)

    y = lambda t, p: np.ones_like(t) * p[0]
    p0 = np.array([2])
    n_skip = 0
    n_take = 11

    non_nan = np.where(~ np.isnan(deltaEs_std_m))
    is_nan = np.where(np.isnan(deltaEs_std_m))

    p_fm, p_std_fm, f_std_fm, return_data_fm = fit(y, considered_T[non_nan][n_skip:n_take]
                                       , deltaEs_m[non_nan][n_skip:n_take]
                                       , deltaEs_std_m[non_nan][n_skip:n_take]
                                       , p0
                                       , statistic=lambda means: dEdec(means)[non_nan][n_skip:n_take]
                                       , data=considered_corrf
                                       , fit_kwargs={"eps1": 1e-5, "eps2": 1e-5, "eps3": 1e-3, "eps4": 1e-3}
                                       , jk_use_central_result=True)
    results_fm.append([p_fm[0], p_std_fm[0]])

    print(i, "fm", p_fm[0], p_std_fm[0])

results_fm = np.array(results_fm)
results_um = np.array(results_um)
results_hm = np.array(results_hm)

with open("results_3st_extraction.csv", "w") as fout:
    print("um", results_um.shape[0], np.mean(results_um[:,0]), np.std(results_um[:,0]), np.mean(results_um[:,1]), np.std(results_um[:,1]), file=fout)
    print("hm", results_hm.shape[0], np.mean(results_hm[:,0]), np.std(results_hm[:,0]), np.mean(results_hm[:,1]), np.std(results_hm[:,1]), file=fout)
    print("fm", results_fm.shape[0], np.mean(results_fm[:,0]), np.std(results_fm[:,0]), np.mean(results_fm[:,1]), np.std(results_fm[:,1]), file=fout)

results = np.array([results_um, results_hm, results_fm])
np.save("results_3st_extraction.npy", results)
