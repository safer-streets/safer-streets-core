import numpy as np
import pandas as pd
from scipy.stats import poisson


def lorenz_curve(
    data_in: pd.DataFrame | pd.Series,
    data_col: str | None = None,
    *,
    weight_col: str | None = None,
    normalise_x: bool = True,
    normalise_y: bool = True,
) -> pd.Series:
    """Full-fat version"""
    if data_in.empty:
        raise ValueError("Input is empty, cannot compute Lorenz curve")
    if not weight_col:
        data = data_in[[data_col]].copy()
        data["unit"] = 1
        weight_col = "unit"
    else:
        data = data_in[[data_col, weight_col]].copy()
    data["order"] = data[data_col] / data[weight_col]
    data = data.sort_values(by=["order", data_col], ascending=False).cumsum().set_index(weight_col, drop=True)[data_col]
    # add origin
    data.loc[0.0] = 0.0
    # return pd.Series(index=1.0 - data.index / data.index.max(), data=data[data_col]).sort_index()
    if normalise_x:
        data = data.set_axis(data.index / data.index.max())
    if normalise_y:
        data = data / data.max()
    return data.sort_index()


def lorenz_baseline_from_poisson(lambda_: float) -> pd.Series:
    # this approach only works for pure Poisson, use the function below for other dists
    kmax = int(1 + lambda_ * 6 * np.sqrt(lambda_))  # rule of thumb for k cutoff
    pmf = pd.Series(poisson(lambda_).pmf(range(kmax)))
    x = np.insert(pmf.cumsum(), 0, 0)
    baseline = pd.Series(index=1 - x[1:], data=1 - x[:-1])
    # baseline = pd.Series(index=(1 - pmf.cumsum()), data=(1 - pmf.cumsum()).shift().values).fillna(1)
    baseline.loc[0.0] = 0.0
    baseline.loc[1.0] = 1.0
    return baseline.sort_index()


def lorenz_baseline_from_pmf(pmf: pd.Series) -> pd.Series:
    lorenz = pd.Series()
    lorenz.loc[1.0] = 1.0

    mean_mixture = sum(k * p for k, p in pmf.items())
    cumulative_prob = 0.0
    cumulative_value_share = 0.0
    for k, p in pmf.items():
        cumulative_prob += p
        cumulative_value_share += (k * p) / mean_mixture
        lorenz.loc[1 - cumulative_prob] = 1 - cumulative_value_share
    return lorenz.sort_index()


def calc_gini(lorenz: pd.Series, *, ref: pd.Series | None = None) -> float:
    gini = (lorenz.index.diff() * lorenz.rolling(2).sum()).sum() / lorenz.index.max() - 1.0
    gini_ref = ((ref.index.diff() * ref.rolling(2).sum()).sum() / ref.index.max() - 1.0) if ref is not None else 0.0
    return gini - gini_ref  # copilot suggests / (1 - gini_ref)


def cosine_similarity(values: pd.DataFrame) -> float:
    # DataFrame ensure indices are consistent. Assumes 2 cols
    col1 = values.iloc[:, 0]
    col2 = values.iloc[:, 1]
    return (col1 @ col2) / np.sqrt((col1 @ col1) * (col2 @ col2))


# translated from https://github.com/virgesmith/demographyMicrosim/
def diversity_coefficient(proportions: pd.Series) -> float:
    # this causes problems if proportions are nan (e.g. when total pop=0)
    # also values outside [0, 1] possible if proportions doesn't sum to 1
    # assert abs(proportions.sum() - 1.0) < 1e-8, f"{proportions=} {proportions.sum()=}"
    n = len(proportions)
    return (1 - proportions @ proportions) * n / (n - 1)
