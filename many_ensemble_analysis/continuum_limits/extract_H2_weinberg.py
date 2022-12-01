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

logging.basicConfig(filename='extract_H2_weinberg.log', level=logging.INFO)
logging.info("Hello, world.")

def hatf_O(cfg, t, a, omega):
    q = cfg[:, t]
    qp = cfg[:, t-1]
    pi = np.pi
    sqrt = np.sqrt
    omega *= 2  # I have a stupid definition of the potential.
                # FIXME: change V(X) to omega**2 * X^2 / 2
    
    return (
        (
            3 / 8 / sqrt(pi) * sqrt(2*pi*a) * (
                (a/2)**(-5/2)
                - (a/2)**(-3/2) * (qp - q)**2 / (a/2)**2
                )
            + (qp - q)**4 / 16 / (a/2)**4
        ) / 4
        + omega**2 / 4 * q**4
        + omega**2 / 2 * q**2 * (
            1 / 4 / sqrt(pi) * (a/2)**(-3/2) * sqrt(2*pi*a)
            - (qp - q)**2 / (a)**2
            )
        - omega**2 * q * (qp - q) / a
        - omega**2 / 2
    )


for n_tau in n_taus:
    delta_t = beta/n_tau
    data = load_from(out_path, str(n_tau) + ".bindata", n_tau, n_markov)
    logging.info(f"base line data size: {data.size * data.itemsize} B")
    logging.info(f"chunk size: {binsize * data.itemsize} B")
    logging.info(f"currently used memory (n_tau: {n_tau})(before loop): {psutil.Process().memory_info().rss} B")


    H2_data = np.zeros(n_bins, dtype=data.dtype)
    for i in range(n_bins):
        chunk = data[binsize * i: binsize * (i + 1)]
        
        H2_chunk = hatf_O(chunk, 0, delta_t, omega)
        H2_mean = np.mean(H2_chunk)
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

    with open("results_H2_weinberg.csv", "a") as fout:
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
