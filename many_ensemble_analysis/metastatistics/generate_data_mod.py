import subprocess
from pathlib import Path
import os
import numpy as np
from multiprocessing import cpu_count
from threading import Thread

bin_path = Path("../../MC/builddir/")
out_path = Path("/glurch/scratch/knd35666/HO_configs_meta_mod/")
n_tau = 40
n_markov = 750000

def produce_qho_data(omega: float, n_tau: int, beta: float, Delta: float, n_markov: int, tinsert: int, Eb: float, seed, fname):
    if(not isinstance(n_tau, int) or n_tau <= 0):
        raise ValueError("n_tau must be int > 0")
        
    executable = bin_path / "deltatH_qho"
    cmd = [executable, str(omega), str(n_tau), str(beta), str(Delta), str(n_markov), str(tinsert), str(Eb), str(fname), str(seed)]
    
    process = subprocess.run(cmd)

omega = 0.5
beta = 10
Delta = 1
tinsert = 0
Eb = 3.4867

seed = lambda i: str(i*2)


def produce_data_or_not(i):
    if(not os.path.exists(out_path / (str(i) + ".bindata"))):
        print("computing for i =", i)
        produce_qho_data(omega, n_tau, beta, Delta, n_markov, tinsert, Eb, seed(i), out_path / (str(i) + ".bindata"))
        with open(out_path / "data.log", "a") as fout:
            print(omega, n_tau, beta, Delta_from_ntau(n_tau), n_markov, tinsert, Eb, seed(i), out_path / (str(i) + ".bindata"), file=fout)
            print(omega, n_tau, beta, Delta_from_ntau(n_tau), n_markov, tinsert, Eb, seed(i), out_path / (str(i) + ".bindata"))

    else:
        print("skipping", out_path / (str(i) + ".bindata"))

threads = [Thread(target=produce_data_or_not, args=(i,)) for i in range(100)]
for t in threads:
    t.start()

for t in threads:
    t.join()

print("done")
