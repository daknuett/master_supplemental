import subprocess
from pathlib import Path
import os
import numpy as np
from multiprocessing import cpu_count
from threading import Thread

bin_path = Path("../../MC/builddir/")

from paths import out_path_unmod as out_path, meta_unmod

n_tau = meta_unmod["n_tau"]
n_markov = meta_unmod["n_markov"]
omega = meta_unmod["omega"]
beta = meta_unmod["beta"]
Delta = meta_unmod["Delta"]

def produce_qho_data(omega: float, n_tau: int, beta: float, Delta: float, n_markov: int, seed, fname):
    if(not isinstance(n_tau, int) or n_tau <= 0):
        raise ValueError("n_tau must be int > 0")
        
    executable = bin_path / "qho"
    cmd = [executable, str(omega), str(n_tau), str(beta), str(Delta), str(n_markov), str(fname), str(seed)]
    
    process = subprocess.run(cmd)


seed = lambda i: str(i*2)


def produce_data_or_not(i):
    if(not os.path.exists(out_path / (str(i) + ".bindata"))):
        print("computing for i =", i)
        produce_qho_data(omega, n_tau, beta, Delta, n_markov, seed(i), out_path / (str(i) + ".bindata"))
        with open(out_path / "data.log", "a") as fout:
            print(omega, n_tau, beta, Delta_from_ntau(n_tau), n_markov, seed(i), out_path / (str(i) + ".bindata"), file=fout)
            print(omega, n_tau, beta, Delta_from_ntau(n_tau), n_markov, seed(i), out_path / (str(i) + ".bindata"))

    else:
        print("skipping", out_path / (str(i) + ".bindata"))

threads = [Thread(target=produce_data_or_not, args=(i,)) for i in range(100)]
for t in threads:
    t.start()

for t in threads:
    t.join()

print("done")
