import numpy as np
import sqlite3

results_gs = np.zeros((100, 2), dtype=float)
results_es = np.zeros((100, 2), dtype=float)

db = sqlite3.connect("result.sqlite3")
cur = db.execute("SELECT ensemble_id, result, std FROM results WHERE method = 'extract_GEVP_q3q5_delta_mod.py' AND result_name = 'gs'")
for (eid, result, std) in cur:
    results_gs[eid][0] = result
    results_gs[eid][1] = std

cur = db.execute("SELECT ensemble_id, result, std FROM results WHERE method = 'extract_GEVP_q3q5_delta_mod.py' AND result_name = 'es'")
for (eid, result, std) in cur:
    results_es[eid][0] = result
    results_es[eid][1] = std
db.close()



results = np.array([results_gs, results_es])
np.save("results_GEVP_q3q5_delta_mod.npy", results)
print("saved results_GEVP_q3q5_delta_mod.npy")

