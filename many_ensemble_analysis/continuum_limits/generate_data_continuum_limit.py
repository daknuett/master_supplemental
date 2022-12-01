import subprocess
from pathlib import Path
import os
import numpy as np
from multiprocessing import cpu_count
from threading import Thread

bin_path = Path("MC/builddir/")
out_path = Path("/glurch/scratch/knd35666/HO_configs_continuum_limit_high_precision/")
n_taus = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 40, 45, 50, 60, 80, 100]
n_markov = 16000000

def produce_qho_data(omega: float, n_tau: int, beta: float, Delta: float, n_markov: int, seed, fname):
    if(not isinstance(n_tau, int) or n_tau <= 0):
        raise ValueError("n_tau must be int > 0")
        
    executable = bin_path / "qho"
    cmd = [executable, str(omega), str(n_tau), str(beta), str(Delta), str(n_markov), str(fname), str(seed)]
    
    process = subprocess.run(cmd)

omega = 0.5
beta = 10
Delta = 1
n_markov = 16000000 

Delta_from_ntau = lambda n: max(Delta + 0.2 - np.log(n) / 10, 0.2)
seed = lambda n_tau: str(n_tau*2)


def produce_data_or_not(n_tau):
    if(not os.path.exists(out_path / (str(n_tau) + ".bindata"))):
        print("computing for ntau =", n_tau)
        produce_qho_data(omega, n_tau, beta, Delta_from_ntau(n_tau), n_markov, seed(n_tau), out_path / (str(n_tau) + ".bindata"))
        with open(out_path / "data.log", "a") as fout:
            print(omega, n_tau, beta, Delta_from_ntau(n_tau), n_markov, seed(n_tau), out_path / (str(n_tau) + ".bindata"), file=fout)
            print(omega, n_tau, beta, Delta_from_ntau(n_tau), n_markov, seed(n_tau), out_path / (str(n_tau) + ".bindata"))

    else:
        print("skipping", out_path / (str(n_tau) + ".bindata"))

threads = [Thread(target=produce_data_or_not, args=(n_tau,)) for n_tau in n_taus]
for t in threads:
    t.start()

for t in threads:
    t.join()

print("done")
