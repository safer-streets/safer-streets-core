from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import chisquare, expon, gamma, lognorm, nbinom, poisson


def poisson_fit(data: pd.Series) -> Any:
    # trivial fit
    return poisson(data.mean())


# dont think this works
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


def nbinom_poisson_gamma(data: pd.Series) -> Any:
    alpha = data.sum()
    scale = 1 / len(data)

    return nbinom(alpha, 1 / (1 + scale))


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


def poisson_pvalue(data: pd.Series) -> float:
    """Calculate chi-squared statistic and return p-value for Poisson goodness-of-fit.

    Args:
        data (pd.Series): Observed count data.

    Returns:
        tuple: Chi-squared statistic and p-value.
    """

    poisson_dist = poisson(data.mean())
    kmax = int(poisson_dist.ppf(1 - np.sqrt(np.finfo(float).eps)))

    pmf = (
        data.value_counts(normalize=True)
        .sort_index()
        .reindex(range(0, kmax + 1), fill_value=0)
        .to_frame(name="observed")
    )
    pmf["expected"] = poisson_dist.pmf(pmf.index)

    return chisquare(pmf.observed, pmf.expected, ddof=0, sum_check=False)[1]
