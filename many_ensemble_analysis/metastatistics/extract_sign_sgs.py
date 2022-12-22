from pathlib import Path
import numpy as np
from itertools import product

import os, fcntl

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



def epsf_H(X, tinsert, omega2, delta_t):
    return 0.5 - (
        (X[:, tinsert + 1] - X[:, tinsert])**2 / 2 / delta_t
        - (X[:, tinsert + 1]**2 + X[:, tinsert]**2) / 2 * omega2 * delta_t
    )

i = os.environ["SLURM_ARRAY_TASK_ID"]

ensemble2 = load_from(out_path_mod, str(i) + ".bindata", n_tau, n_markov)
sign_data = np.sign(epsf_H(ensemble2, tinsert, omega, delta_t) - delta_t*Eb)

out_name = "results_sign_sgs.tsv"

with open(out_name, "a") as fout:
    fcntl.flock(fout.fileno(), fcntl.LOCK_EX)
    fout.seek(0, 2)
    print(i, n_markov, np.mean(sign_data), np.var(sign_data), file=fout)
    fcntl.flock(fout.fileno(), fcntl.LOCK_UN)
