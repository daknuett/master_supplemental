
h_weinberg_code = '''
def hatf_O(cfg, t, a, omega):
    V = omega * cfg[:,t]**2
    T = (-(cfg[:, t-1] - cfg[:,t])**2 / a**2 * numpy.sqrt(2*numpy.pi/a)
         + numpy.sqrt(numpy.pi) / 2 * (a/2)**(-3/2))
    return V + numpy.sqrt(2*numpy.pi*a) * T / 4 / numpy.pi 

def observable_doit(data, args):
    delta_t, omega = args
    H_data = hatf_O(data, 0, delta_t, omega)
    H_mean = numpy.mean(H_data)
    H_std = numpy.std(H_data) / numpy.sqrt(H_data.shape[0])
    return H_mean, H_std
'''

h_weinberg_requirements = ["numpy"]


h_lehner_code = '''
def S(Xi, omega2, delta_t, is_euclidean=False):
    if(len(Xi.shape) == 1):
        axis = 0
        idx = slice(None, -1, 1)
    else:
        axis = 1
        idx = slice(None, None, 1), slice(None, -1, 1)
    T = 0.5 * numpy.sum(((Xi - numpy.roll(Xi, -1, axis=axis))**2)[idx], axis=axis)
    #V = 0.5 * omega2 * numpy.sum((Xi**2 + numpy.roll(Xi, -1, axis=axis)**2)[idx], axis=axis)
    V = omega2 * numpy.sum((numpy.roll(Xi, -1, axis=axis)**2)[idx], axis=axis)

    if(is_euclidean):
        return T / delta_t + V * delta_t
    return T / delta_t - V * delta_t

def S_inlet(Xi, omega2, delta_t, ifrom, ito, is_euclidean=False):
    if(len(Xi.shape) == 1):
        return S(Xi[ifrom:ito], omega2, delta_t, is_euclidean=is_euclidean)
    return S(Xi[:, ifrom:ito], omega2, delta_t, is_euclidean=is_euclidean)


def observable_doit(data, args):
    delta_t, omega = args 
    Z_m_Z_mean = np.average((.5 - S_inlet(data, omega, delta_t, 0, 0+2)))
    Z_m_Z_std = np.std((.5 - S_inlet(data, omega, delta_t, 0, 0+2))) / np.sqrt(data.shape[0])
    E0_mean = Z_m_Z_mean / delta_t
    E0_std = Z_m_Z_std / delta_t
    return E0_mean, E0_std
'''
h_lehner_requirements = ["numpy"]

various_observables_code = '''
def hatf_H(cfg, t, a, omega):
    V = omega * cfg[:,t]**2
    T = (-(cfg[:, t-1] - cfg[:,t])**2 / a**2 * numpy.sqrt(2*numpy.pi/a)
         + numpy.sqrt(numpy.pi) / 2 * (a/2)**(-3/2))
    return V + numpy.sqrt(2*numpy.pi*a) * T / 4 / numpy.pi 


n_bins = 8000

def make_bins(data, nbins):
    bin_axis = len(data.shape) - 1
    total_items = data.shape[bin_axis]
    binsize = total_items // nbins
    if(total_items % nbins):
        raise ValueError(f"number of items({total_items}) must be multiple of nbins({nbins})")
    other_dimensions = data.shape[:-1]
    
    
    return numpy.mean(data.reshape(*other_dimensions, -1, binsize), axis=bin_axis + 1)

def jackknife2_std(data, statistic, *params, data_axis=1, statistic_axis=0, masked_ok=True):
    """
    This is the full-featured standard deviation error compared to
    ``jackknife_std``. Unlike ``jackknife_std`` this function passes full
    samples instead of means to the ``statistic``.

    This function is slower but can be used for more complex statistics.

    See eqs. (1.4), (1.5)::
        @article{10.1214/aos/1176345462,
        author = {B. Efron and C. Stein},
        title = {{The Jackknife Estimate of Variance}},
        volume = {9},
        journal = {The Annals of Statistics},
        number = {3},
        publisher = {Institute of Mathematical Statistics},
        pages = {586 -- 596},
        keywords = {$U$ statistics, ANOVA decomposition, bootstrap, jackknife, variance estimation},
        year = {1981},
        doi = {10.1214/aos/1176345462},
        URL = {https://doi.org/10.1214/aos/1176345462}
        }
    """

    N = data.shape[data_axis]
    if(data_axis == 0):
        def slc(i):
            return i
    else:
        def slc(i):
            lst = [slice(None, None, None)]*data_axis
            return tuple(lst + [i])

    masked = numpy.ma.array(data, mask=False)

    # FIXME: This is terribly slow. 
    # Can we improve the performance by parallelism?
    # Are there other fancy ways to improve performance?
    jackknife_samples = []
    for i in range(N):
        masked.mask[slc(i)] = True
        if(masked_ok):
            jackknife_samples.append(statistic(masked, *params))
        else:
            jackknife_samples.append(statistic(numpy.array(masked), *params))
        masked.mask[slc(i)] = False
    jackknife_samples = numpy.array(jackknife_samples)

    jackmean = numpy.mean(jackknife_samples, axis=statistic_axis)

    return numpy.sqrt(numpy.sum((jackknife_samples - jackmean)**2, axis=statistic_axis) * (N - 1) / N)

def observable_doit(data, args):
    delta_t, omega, n_tau = args 

    data = data[::4,:]

    binsize = data.shape[0] / n_bins

    H_data = hatf_H(data, 0, delta_t, omega)
    H_data = make_bins(H_data, n_bins)
    variance_H = numpy.var(H_data) * binsize
    variance_H_err = jackknife2_std(H_data, lambda x: numpy.var(x)*binsize , data_axis=0)
    del(H_data)

    data_q = data[:,0]
    variance_q = numpy.var(data_q) 
    data_q = make_bins(data_q, n_bins)
    #variance_q = numpy.var(data_q) * binsize
    variance_q_err = jackknife2_std(data_q, lambda x: numpy.var(x)*binsize , data_axis=0)

    data_q2 = data[:,0]**2
    #variance_q2 = numpy.var(data_q2) 
    data_q2 = make_bins(data_q2, n_bins)
    variance_q2 = numpy.var(data_q2) * binsize
    variance_q2_err = jackknife2_std(data_q2, lambda x: numpy.var(x)*binsize , data_axis=0)

    data_qqp = data[:,0] - data[:,1]
    data_qqp = make_bins(data_qqp, n_bins)
    variance_qqp = numpy.var(data_qqp) * binsize
    variance_qqp_err = jackknife2_std(data_qqp, lambda x: numpy.var(x)*binsize , data_axis=0)

    data_qqp2 = (data[:,0] - data[:,1])**2
    data_qqp2 = make_bins(data_qqp2, n_bins)
    variance_qqp2 = numpy.var(data_qqp2) * binsize
    variance_qqp2_err = jackknife2_std(data_qqp2, lambda x: numpy.var(x)*binsize , data_axis=0)
    return (numpy.array([n_tau, delta_t, variance_H, variance_q, variance_q2, variance_qqp, variance_qqp2])
            , numpy.array([0, 0, variance_H_err, variance_q_err, variance_q2_err, variance_qqp_err, variance_qqp2_err]))
'''

various_observables_requirements = ["numpy"]

import sqlite3
from db import add_observable_extractor

conn = sqlite3.connect("database.sqlite3")

#add_observable_extractor(conn, h_weinberg_code, h_weinberg_requirements, 0, None)
#add_observable_extractor(conn, h_lehner_code, h_lehner_requirements, 0, None)

c = conn.execute("SELECT rowid FROM encodings")
encoding = c.fetchone()[0]
add_observable_extractor(conn, various_observables_code, various_observables_requirements, True, encoding)

conn.commit()
conn.close()

