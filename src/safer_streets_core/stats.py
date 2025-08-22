from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import expon, gamma, lognorm, nbinom, poisson


def poisson_fit(data: pd.Series) -> Any:
    # trivial fit
    return poisson(data.mean())


def nbinom_fit(data: pd.Series) -> Any:
    # Histogram of counts (wont contain any zeros)
    values, counts_ = np.unique(data, return_counts=True)

    def nbinom_pdf(x, n, p):
        return nbinom.pmf(x, n, p) * data.size

    # make initial guesses
    mean = data.mean()
    var = data.var()
    p0 = mean / var if var > mean else 0.5
    n0 = mean**2 / (var - mean) if var > mean else 1.0
    # fit
    popt, _ = curve_fit(nbinom_pdf, values, counts_, p0=[n0, p0], bounds=([1e-9, 1e-9], [np.inf, 1 - 1e-9]))
    n, p = popt
    # mu = n * (1 - p) / p
    return nbinom(n, p)


def gamma_fit(data: pd.Series) -> Any:
    # # Estimate k (shape) and theta (scale) using Method of Moments formulas
    # mean = data.mean()
    # var = data.var(ddof=1)
    # k_mom = mean**2 / var
    # theta_mom = var / mean

    # use maximum likelihood estimation (MLE) to fit the gamma distribution
    a_mle, loc_mle, scale_mle = gamma.fit(data)
    return gamma(a_mle, loc=loc_mle, scale=scale_mle)


def exponential_fit(data: pd.Series) -> Any:
    loc, scale = expon.fit(data)  # fit with fixed location at 0
    return expon(loc=loc, scale=scale)


def lognorm_fit(data: pd.Series) -> Any:
    shape, loc, scale = lognorm.fit(data)  # fit with fixed location at 0
    return lognorm(shape, loc=loc, scale=scale)
