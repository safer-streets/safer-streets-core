from itertools import zip_longest

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import f1_score


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
    # remove potentially duplicated values when x ~ double epsilon (happens with large lambda)
    return baseline.sort_index()[~baseline.sort_index().index.duplicated()]


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


def calc_gini0(lambda_: float) -> float:
    """Gini value for "null hypothesis" (i.i.d. Poisson)"""
    lorenz = lorenz_baseline_from_poisson(lambda_)
    gini = (lorenz.index.diff() * lorenz.rolling(2).sum()).sum() / lorenz.index.max() - 1.0
    return gini


def calc_overdispersion(data: pd.Series) -> float:
    # coeff. of variation CV (std/mean) for Poisson is 1/sqrt(lambda)
    # return the difference between the CV of the data and the Poisson value
    lambda_ = data.mean()
    return data.std() / lambda_ - 1 / np.sqrt(lambda_)


def cosine_similarity(values: pd.DataFrame) -> float:
    # DataFrame ensure indices are consistent. Assumes 2 cols
    col1 = values.iloc[:, 0]
    col2 = values.iloc[:, 1]
    return (col1 @ col2) / np.sqrt((col1 @ col1) * (col2 @ col2))


def _spearman_rank_correlation_impl(diff: pd.Series) -> float:
    n = len(diff)
    return 1 - 6 * (diff**2).sum() / (n * (n * n - 1))


def spearman_rank_correlation(counts: pd.DataFrame) -> float:
    # rank based on counts (default method "average" treats 2 tied at 3rd place as 3.5)
    ranks = counts.apply(lambda col: col.rank(ascending=False))
    # DataFrame ensure indices are consistent. Assumes 2 cols
    return _spearman_rank_correlation_impl(ranks.iloc[:, 0] - ranks.iloc[:, 1])


def spearman_rank_correlation_matrix(counts: pd.DataFrame) -> npt.NDArray:
    """
    Calculate the Spearman rank correlation for each pair of columns in a DataFrame.
    Returns a symmetric matrix with correlations.
    """
    assert counts.index.is_unique, "Index must be unique for correlation calculation"
    assert not counts.empty, "DataFrame must not be empty"

    # Initialize a square matrix for correlations
    n = len(counts.columns)
    correlations = np.eye(n)

    ranks = counts.apply(lambda col: col.rank(ascending=False))

    for i in range(n):
        for j in range(i):
            correlations[i, j] = correlations[j, i] = _spearman_rank_correlation_impl(
                ranks.iloc[:, i] - ranks.iloc[:, j]
            )
    return correlations


def rank_biased_overlap(counts: pd.DataFrame, decay: float = 0.9) -> float:
    """
    Slightly limited home-made rank-biased overlap score.
    Input is a 2-col dataframe (ensuring consistent indices)
    This means there will always be a positive score due to the final term (all vs all)
    """

    # rank based on counts (default method "average" treats 2 tied at 3rd place as 3.5)
    ranks = counts.apply(lambda col: col.rank(ascending=False))

    left_sets = tuple(set(group.index) for _, group in ranks.iloc[:, 0].groupby(ranks.iloc[:, 0]))
    right_sets = tuple(set(group.index) for _, group in ranks.iloc[:, 1].groupby(ranks.iloc[:, 1]))

    num = 0.0
    den = 0.0
    union = set()
    intersection = set()
    for i, (left, right) in enumerate(zip_longest(left_sets, right_sets, fillvalue=set())):
        # enumerate any already encountered in the other set
        inter1 = (union & left) | (union & right)
        # now update the union...
        union |= left | right
        # ...and the intersection
        intersection |= (left & right) | inter1
        num += decay**i * len(intersection) / len(union)
        den += decay**i
    return num / den


def threshold_f1(pred_target: pd.DataFrame, threshold: float) -> float:
    ranked = pd.concat([col.sort_values().cumsum() / col.sum() for _, col in pred_target.items()], axis=1).sort_index()
    result = ranked > 1 - threshold
    return f1_score(result.iloc[:, 0], result.iloc[:, 1])


# translated from https://github.com/virgesmith/demographyMicrosim/
def diversity_coefficient(proportions: pd.Series) -> float:
    # this causes problems if proportions are nan (e.g. when total pop=0)
    # also values outside [0, 1] possible if proportions doesn't sum to 1
    # assert abs(proportions.sum() - 1.0) < 1e-8, f"{proportions=} {proportions.sum()=}"
    n = len(proportions)
    return (1 - proportions @ proportions) * n / (n - 1)
