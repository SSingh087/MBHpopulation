import numpy as np
from scipy.stats import beta, norm
from utils import Plotting


def universal_EMRI_rate(tau):
    # this should be replaced by the values that Luca will provide, but for now we can use similar looking distributions to test the code
    # From arXiv:2205.06277v1 Fig 6 
    a, b = 2.0, 5.0
    return beta.pdf(tau, a, b)


def universal_TDE_rate(tau):
    # this should be replaced by the values that Luca will provide, but for now we can use similar looking distributions to test the code
    # From arXiv:2205.06277v1 Fig 6 
    mu, sigma = 1e-7, 0.05
    a, b = 2.0, 5.0
    return norm.pdf(tau, mu, sigma) + beta.pdf(tau, a, b)


def cdf_from_pdf(x, pdf):
    
    x = np.asarray(x)

    pdf = np.asarray(pdf)

    area = np.trapezoid(pdf, x)
    pdf = pdf / area

    cdf = np.empty_like(pdf)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(x))
    # Numerical guard : enforce last value to be exactly 1
    cdf = np.clip(cdf, 0.0, 1.0)
    return cdf

def draw_samples(x, cdf, size=None):
    u = np.random.uniform(0.0, 1.0, size=size)
    u = np.clip(u, cdf.min(), cdf.max())
    return np.interp(u, cdf, x)

def get_samples(tau, kind='EMRI', size=1000):
    if kind == 'EMRI':
        pdf = universal_EMRI_rate(tau)
        cdf = cdf_from_pdf(tau, pdf)
    else:
        pdf = universal_TDE_rate(tau)
        cdf = cdf_from_pdf(tau, pdf)
    samples = draw_samples(tau, cdf, size=size)
    return samples

tau = np.linspace(0, 1, 1000)
Plotting.plot_rate_evolution(tau, universal_EMRI_rate(tau), universal_TDE_rate(tau))