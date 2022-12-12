import subprocess
from pathlib import Path
import os
import numpy as np
from multiprocessing import cpu_count
from threading import Thread

#from paths import out_path_mod as out_path, meta_mod
from paths import out_path_mod_gs as out_path, meta_mod_gs as meta_mod

bin_path = Path("../../MC/builddir/")

def produce_qho_data(omega: float, n_tau: int, beta: float, Delta: float, n_markov: int, tinsert: int, Eb: float, seed, fname):
    if(not isinstance(n_tau, int) or n_tau <= 0):
        raise ValueError("n_tau must be int > 0")
        
    executable = bin_path / "deltatH_qho"
    cmd = [executable, str(omega), str(n_tau), str(beta), str(Delta), str(n_markov), str(tinsert), str(Eb), str(fname), str(seed)]
    
    process = subprocess.run(cmd)

n_tau = meta_mod["n_tau"]
n_markov = meta_mod["n_markov"]
omega = meta_mod["omega"]
beta = meta_mod["beta"]
Delta = meta_mod["Delta"]
tinsert = meta_mod["tinsert"]
Eb = meta_mod["Eb"]

seed = lambda i: str(i*2)


def produce_data_or_not(i):
    if(not os.path.exists(out_path / (str(i) + ".bindata"))):
        print("computing for i =", i)
        produce_qho_data(omega, n_tau, beta, Delta, n_markov, tinsert, Eb, seed(i), out_path / (str(i) + ".bindata"))
        with open(out_path / "data.log", "a") as fout:
            print(omega, n_tau, beta, Delta, n_markov, tinsert, Eb, seed(i), out_path / (str(i) + ".bindata"), file=fout)
            print(omega, n_tau, beta, Delta, n_markov, tinsert, Eb, seed(i), out_path / (str(i) + ".bindata"))

    else:
        print("skipping", out_path / (str(i) + ".bindata"))

threads = [Thread(target=produce_data_or_not, args=(i,)) for i in range(100)]
for t in threads:
    t.start()

for t in threads:
    t.join()

print("done")
