from pathlib import Path
import numpy as np
from itertools import product

import sys
sys.path.append("../")
from load_data import load_from

sys.path.append("../../statistics")

from statistic.statistic import make_bins, jackknife_std, tauint

out_path = Path("/glurch/scratch/knd35666/HO_configs_autocorrelation_stuff/")

omega = 0.5
Delta = 1
beta = 10 
n_tau = 40
delta_t = beta/n_tau
n_markov = 750000

for i in range(100):
    data = load_from(out_path, str(i) + ".bindata", n_tau, n_markov)
