from pathlib import Path
import numpy as np
import sys
import psutil
import logging

from data_list import n_taus, out_path, n_markov

import sys
sys.path.append("../")
from load_data import load_from

sys.path.append("../../statistics")
from statistic.statistic import make_bins, jackknife_std


omega = 0.5
beta = 10
Delta = 1
n_bins = 200
binsize = n_markov // n_bins

logging.basicConfig(filename='extract_H2_lehner.log', level=logging.INFO)
logging.info("Hello, world.")

def S(Xi, omega2, delta_t, is_euclidean=False):
    if(len(Xi.shape) == 1):
        axis = 0
        idx = slice(None, -1, 1)
    else:
        axis = 1
        idx = slice(None, None, 1), slice(None, -1, 1)
    T = 0.5 * np.sum(((Xi - np.roll(Xi, -1, axis=axis))**2)[idx], axis=axis)
    #V = 0.5 * omega2 * np.sum((Xi**2 + np.roll(Xi, -1, axis=axis)**2)[idx], axis=axis)
    #V = omega2 * np.sum((np.roll(Xi, -1, axis=axis)**2)[idx], axis=axis)
    V = omega2 * np.sum((np.roll(Xi, -1, axis=axis)**2)[idx], axis=axis)
    
    if(is_euclidean):
        return T / delta_t + V * delta_t
    return T / delta_t - V * delta_t

def S_inlet(Xi, omega2, delta_t, ifrom, ito, is_euclidean=False):
    if(len(Xi.shape) == 1):
        return S(Xi[ifrom:ito], omega2, delta_t, is_euclidean=is_euclidean)
    return S(Xi[:, ifrom:ito], omega2, delta_t, is_euclidean=is_euclidean)

for n_tau in n_taus:
    delta_t = beta/n_tau
    data = load_from(out_path, str(n_tau) + ".bindata", n_tau, n_markov)
    logging.info(f"base line data size: {data.size * data.itemsize} B")
    logging.info(f"chunk size: {binsize * data.itemsize} B")
    logging.info(f"currently used memory (n_tau: {n_tau})(before loop): {psutil.Process().memory_info().rss} B")


    H2_data = np.zeros(n_bins, dtype=data.dtype)
    for i in range(n_bins):
        chunk = data[binsize * i: binsize * (i + 1)]
        
        H2_chunk = ((.5 - S_inlet(chunk, omega, delta_t, 0, 0+2))**2
                            + (.5 - S_inlet(chunk, omega, delta_t, 0, 0+2))
                            - S_inlet(chunk, omega, delta_t, 0, 0+2, is_euclidean=True))
        H2_mean = np.mean(H2_chunk) / delta_t**2
        H2_data[i] = H2_mean
        logging.info(f"currently used memory (chunk: {i})(after computation in loop): {psutil.Process().memory_info().rss} B")
        before_del = psutil.Process().memory_info().rss
        del(chunk)
        del(H2_chunk)
        del(H2_mean)
        after_del = psutil.Process().memory_info().rss
        logging.info(f"currently used memory (chunk: {i})(after cleanup in loop): {psutil.Process().memory_info().rss} B")
        logging.info(f"free'd during cleanup (chunk: {i}): {before_del - after_del} B")

    H2_mean = np.mean(H2_data)
    H2_std = np.std(H2_data) / np.sqrt(H2_data.shape[0])
    E0_expect = 0.5
    tension = (H2_mean - E0_expect**2) / (H2_std)

    with open("results_H2_lehner.csv", "a") as fout:
        print(n_tau, H2_mean, H2_std, beta, file=fout)
    before_del = psutil.Process().memory_info().rss
    del(data)
    del(H2_data)
    del(H2_mean)
    del(H2_std)
    del(tension)
    after_del = psutil.Process().memory_info().rss
    logging.info(f"currently used memory (n_tau: {n_tau})(cleanup after loop): {psutil.Process().memory_info().rss} B")
    logging.info(f"free'd during cleanup (n_tau: {n_tau}): {before_del - after_del} B")

