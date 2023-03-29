import sqlite3, json
from itertools import product

conn = sqlite3.connect("database.sqlite3")

c = conn.execute("SELECT rowid, meta FROM ensembles WHERE collection=2")
ensembles = c.fetchall()
rowids = [ens[0] for ens in ensembles]
metas = [json.loads(ens[1]) for ens in ensembles]
delta_ts = [float(m["beta"]) / float(m["n_tau"]) for m in metas]
n_taus = [int(m["n_tau"]) for m in metas]



extractors = [3]
onames = ["variances"]

requests = [(i, ex[0], ens[0], str((ens[1], 0.5, ens[2])), ex[1]) for i, (ens, ex) in enumerate(product(zip(rowids, delta_ts, n_taus), zip(extractors, onames)))]


conn.executemany("INSERT INTO observable_requests VALUES(?, ?, ?, ?, ?)", requests)
conn.commit()
conn.close()

