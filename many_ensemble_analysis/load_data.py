from pathlib import Path
import os
import numpy as np

def discover_data(path
                  , prefix="QHO_data_"
                  , converters=[float, int, float, float, int]):
    data_directories = [name for name in os.listdir(path)
                                     if (name.startswith(prefix)
                                         and os.path.isdir(path / name))]
    
    pl = len(prefix)
    dirs_and_meta = [(name, [c(v) for c,v in zip(converters, name[pl:].split("_"))])
                         for name in data_directories]
    return dirs_and_meta

def checkfor_data(path, *params, prefix="QHO_data_"):
    dirname = prefix + "_".join(str(param) for param in params)
    return os.path.isdir(path / dirname), dirname

def discover_data_files(rootpath, path, extension=".bindata"):
    pth = rootpath / path
    return [name for name in os.listdir(pth) if (os.path.isfile(pth/name)
                                                 and name.endswith(extension))]

def get_data_files(rootpath, *params, prefix="QHO_data_", extension=".bindata"):
    exists, pname = checkfor_data(rootpath
                                  , *params
                                  , prefix=prefix)
    if(not exists):
        print("no data found at", rootpath / pname)
        return  []
    dfiles = discover_data_files(rootpath, pname, extension=extension)
    print("found", len(dfiles), "datafiles in", rootpath / pname)
    return rootpath / pname, list(sorted(dfiles))

def load_from(path, filename, n_tau, n_markov):
    fname = path / filename
    data = np.fromfile(str(fname))
    data = data.reshape((n_markov, n_tau))
    return data
