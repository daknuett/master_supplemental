import sqlite3 
import os
import importlib
import pathlib
import json 
import numpy as np

DEFAULT_DB_NAME = "database.sqlite3"

if "EXTRACTION_DB" in os.environ:
    dbfname = os.environ["EXTRACTION_DB"]
else:
    dbfname = DEFAULT_DB_NAME


# Fetch request information
conn = sqlite3.connect(DEFAULT_DB_NAME)
conn.execute("PRAGMA busy_timeout = 30000")

i = os.environ["SLURM_ARRAY_TASK_ID"]

c = conn.execute("SELECT extractor, ensemble, params, name FROM observable_requests WHERE requestno = ?", (i,))
(extractor, ensemble, params, name) = c.fetchone()

c = conn.execute("SELECT code, returns_array, array_encoding FROM observable_extractors WHERE rowid=?", (extractor,))
(code, returns_array, array_encoding) = c.fetchone()

if(returns_array):
    c = conn.execute("SELECT code FROM encodings WHERE rowid=?", (array_encoding,))
    encoding_code = c.fetchone()[0]

    c = conn.execute("SELECT name FROM py_modules INNER JOIN encoding_module_requirements ON py_modules.rowid = encoding_module_requirements.module WHERE encoding_module_requirements.encoding=?", (array_encoding,))
    modules_for_encoding = [v[0] for v in c.fetchall()]

c = conn.execute("SELECT name FROM py_modules INNER JOIN observable_extractor_requirements ON py_modules.rowid = observable_extractor_requirements.module WHERE observable_extractor_requirements.observable_extractor=?", (extractor,))
modules_for_code = [v[0] for v in c.fetchall()]

c = conn.execute("SELECT fname, meta, collection FROM ensembles WHERE rowid=?", (ensemble,))
fname, meta, collection = c.fetchone()

c = conn.execute("SELECT path FROM ensemble_collections WHERE rowid=?", (collection,))
collection_path = c.fetchone()[0]

conn.close()
# Information fetching is done.



# Set up environment.
for module in modules_for_code:
    #importlib.import_module(module)
    exec("import " + module)
exec(code)

if(returns_array):
    for module in modules_for_encoding:
        #importlib.import_module(module)
        exec("import " + module)

    exec(encoding_code)

collection_path = pathlib.Path(collection_path)

ensemble_path = collection_path / fname

print("FOUND request", i, "on ensemble", ensemble_path)


meta = json.loads(meta)

# Compute result
ensemble_data = np.fromfile(ensemble_path).reshape((int(meta["n_markov"]), int(meta["n_tau"])))
mean, std = observable_doit(ensemble_data, eval(params))

# Store result

conn = sqlite3.connect(dbfname, detect_types=sqlite3.PARSE_DECLTYPES)
conn.execute("PRAGMA busy_timeout = 30000")

if not returns_array:
    conn.execute("INSERT INTO simple_results VALUES(?, ?, ?, ?, ?)", (ensemble, name, params, float(mean), float(std)))
else:
    conn.execute("INSERT INTO array_results VALUES(?, ?, ?, ?, ?, ?)", (ensemble, name, params, array_encoding, mean, std))

conn.commit()
conn.close()

