from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import lognorm, nbinom, poisson


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


def lognorm_fit(data: pd.Series) -> Any:
    logdata = np.log(data[data > 0])  # log-transform data, ignoring zeros
    return lognorm(logdata.std(), loc=logdata.mean(), scale=np.exp(logdata.mean()))
