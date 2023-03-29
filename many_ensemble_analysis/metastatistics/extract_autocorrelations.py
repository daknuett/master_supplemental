from pathlib import Path
import numpy as np
from itertools import product

import os
import sqlite3
import base64, io




array_converter = '''
def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr, allow_pickle=False)
    out.seek(0, 0)
    return sqlite3.Binary(base64.b64encode(out.read()))

def convert_array(text):
    fin = io.BytesIO(base64.b64decode(text))
    return np.load(fin)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)
'''
exec(array_converter)



import sys
sys.path.append("../")
from load_data import load_from

sys.path.append("../../statistics")

from statistic.statistic import make_bins, jackknife_std, tauint

from paths import out_path_unmod_lp as out_path, meta_unmod_lp as meta_unmod


def c11(X, s):
    return np.array([X[:, s] * X[:, s + t]
                     for t in range(len(X[0]))])

def c33(X, s):
    return np.array([X[:, s]**3 * X[:, s + t]**3
                     for t in range(len(X[0]))])

def hatf_H(cfg, t, a, omega):
    V = omega * cfg[:,t]**2
    T = (-(cfg[:, t-1] - cfg[:,t])**2 / a**2 * np.sqrt(2*np.pi/a)
         + np.sqrt(np.pi) / 2 * (a/2)**(-3/2))
    return V + np.sqrt(2*np.pi*a) * T / 4 / np.pi 


omega = meta_unmod["omega"]
Delta = meta_unmod["Delta"]
beta = meta_unmod["beta"]
n_tau = meta_unmod["n_tau"]
n_markov = meta_unmod["n_markov"]

delta_t = beta/n_tau

i = os.environ["SLURM_ARRAY_TASK_ID"]

data = load_from(out_path, str(i) + ".bindata", n_tau, n_markov)

    
H_data = hatf_H(data, 0, delta_t, omega)
tauint_H = tauint(H_data)
del(H_data)

tauint_q2 = tauint(data[:,0]**2)
tauint_q4 = tauint(data[:,0]**4)

c11_data = c11(data, 0)
tauints_11 = np.array([tauint(c) for c in c11_data])
del(c11_data)

c33_data = c33(data, 0)
tauints_33 = np.array([tauint(c) for c in c33_data])
del(c33_data)


db = sqlite3.connect("result.sqlite3", detect_types=sqlite3.PARSE_DECLTYPES)

# method TEXT, result_name TEXT, ensemble_id INT, ensemble_collection TEXT, result NUM, std NUM
cur = db.execute("PRAGMA busy_timeout = 30000")
cur.execute("CREATE TABLE IF NOT EXISTS array_results(method TEXT, result_name TEXT, ensemble_id INT, ensemble_collection TEXT, encoding TEXT, result array, std array)")
db.commit()

print(tauints_11)
print(type(tauints_11))
print(tauints_11.dtype)
print(tauints_33)
print(type(tauints_33))
print(tauints_33.dtype)

print(i)
if(int(i) == 0):
    np.save("tauint_c11.npy", tauints_11)
    print("saved tauint_c11.npy")

cur.execute(" INSERT INTO results VALUES(?, ?, ?, ?, ?, ?)", ("extract_autocorrelations.py", "tauint_H", i, str(out_path), tauint_H, None))
cur.execute(" INSERT INTO results VALUES(?, ?, ?, ?, ?, ?)", ("extract_autocorrelations.py", "tauint_q2", i, str(out_path), tauint_q2, None))
cur.execute(" INSERT INTO results VALUES(?, ?, ?, ?, ?, ?)", ("extract_autocorrelations.py", "tauint_q4", i, str(out_path), tauint_q4, None))
cur.execute(" INSERT INTO array_results VALUES(?, ?, ?, ?, ?, ?, ?)", ("extract_autocorrelations.py", "tauint_c11", i, str(out_path), array_converter, tauints_11, np.zeros_like(tauints_11)))
cur.execute(" INSERT INTO array_results VALUES(?, ?, ?, ?, ?, ?, ?)", ("extract_autocorrelations.py", "tauint_c33", i, str(out_path), array_converter, tauints_33, np.zeros_like(tauints_33)))
db.commit()
db.close()
print("done")
