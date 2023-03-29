import sqlite3
import base64 
import io
import hashlib

import numpy as np


default_array_converter = ('''
def adapt_array(arr):
    out = io.BytesIO()
    numpy.save(out, arr, allow_pickle=False)
    out.seek(0, 0)
    return sqlite3.Binary(base64.b64encode(out.read()))

def convert_array(text):
    fin = io.BytesIO(base64.b64decode(text))
    return numpy.load(fin)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)
''', ["io", "base64", "numpy", "sqlite3"])


def db_init(connection: sqlite3.Connection):
    cursor = connection.cursor()

    cursor.execute("PRAGMA busy_timeout = 30000")

    cursor.execute("CREATE TABLE IF NOT EXISTS encodings(tag TEXT, code TEXT, UNIQUE(tag))")
    cursor.execute("CREATE TABLE IF NOT EXISTS py_modules(name TEXT PRIMARY KEY)")
    cursor.execute("CREATE TABLE IF NOT EXISTS encoding_module_requirements(encoding INT, module INT)")

    cursor.execute("CREATE TABLE IF NOT EXISTS ensemble_collections(path TEXT, UNIQUE(path))")
    cursor.execute("CREATE TABLE IF NOT EXISTS ensembles(collection INT, fname TEXT, meta TEXT)")

    cursor.execute("CREATE TABLE IF NOT EXISTS simple_results(ensemble INT, name TEXT, params TEXT, value REAL, error REAL)")
    cursor.execute("CREATE TABLE IF NOT EXISTS array_results(ensemble INT, name TEXT, params TEXT, encoding INT, value array, error array)")

    cursor.execute("CREATE TABLE IF NOT EXISTS observable_extractors(tag TEXT PRIMARY KEY, code TEXT, returns_array INT, array_encoding INT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS observable_extractor_requirements(observable_extractor INT, module INT)")

    cursor.execute("CREATE TABLE IF NOT EXISTS observable_requests(requestno INT, extractor INT, ensemble INT, params TEXT, name TEXT)")

    connection.commit()

def register_array_converter(connection: sqlite3.Connection, array_converter: tuple):
    """
    Inserts the encoding ``(code, [requirements])`` into the 
    database. Returns the encoding_id.
    """
    cursor = connection.cursor()
    
    code, requirements = array_converter
    tag = hashlib.sha256(code.encode("UTF-8")).hexdigest()

    cursor.executemany("INSERT OR IGNORE INTO py_modules VALUES(?)", [(r,) for r in requirements])

    def fetch_module_id(module_name):
        c = cursor.execute("SELECT rowid FROM py_modules WHERE name=?", (module_name,))
        return c.fetchone()[0]

    requirement_ids = [fetch_module_id(m) for m in requirements]

    cursor.execute("INSERT INTO encodings VALUES(?, ?)", (tag, code))
    c = cursor.execute("SELECT rowid FROM encodings WHERE tag=?", (tag,))
    encoding_id = c.fetchone()[0]

    for rid in requirement_ids:
        cursor.execute("INSERT INTO encoding_module_requirements VALUES(?, ?)", (encoding_id, rid))
    
    connection.commit()
    return encoding_id

def register_ensemble_collection(connection: sqlite3.Connection, path):
    cursor = connection.cursor()
    
    cursor.execute("INSERT INTO ensemble_collections VALUES(?)", (path,))
    c = cursor.execute("SELECT rowid FROM ensemble_collections WHERE path=?", (path,))
    eid = c.fetchone()[0]

    connection.commit()
    return eid

def register_ensemble(connection: sqlite3.Connection, ensemble_id: int, fname: str, meta: str):
    cursor = connection.cursor()
    cursor.execute("INSERT INTO ensembles VALUES(?)", (ensemble_id, fname, meta))
    c = cursor.execute("SELECT rowid FROM ensembles WHERE collection=? AND fname=?", (ensemble_id, fname))
    cid = c.fetchone()[0]

    connection.commit()
    return cid
    
def add_observable_extractor(connection: sqlite3.Connection, code: str, requirements, returns_array, array_encoding):
    cursor = connection.cursor()
    tag = hashlib.sha256(code.encode("UTF-8")).hexdigest()

    cursor.executemany("INSERT OR IGNORE INTO py_modules VALUES(?)", [(r,) for r in requirements])

    def fetch_module_id(module_name):
        c = cursor.execute("SELECT rowid FROM py_modules WHERE name=?", (module_name,))
        return c.fetchone()[0]

    requirement_ids = [fetch_module_id(m) for m in requirements]

    cursor.execute("INSERT INTO observable_extractors VALUES(?, ?, ?, ?)", (tag, code, returns_array, array_encoding))
    c = cursor.execute("SELECT rowid FROM observable_extractors WHERE tag=?", (tag,))
    extractor_id = c.fetchone()[0]

    for rid in requirement_ids:
        cursor.execute("INSERT INTO observable_extractor_requirements VALUES(?, ?)", (extractor_id, rid))
    
    connection.commit()
    return extractor_id


