import numpy as np
import pandas as pd
import pytest

from safer_streets_core.measures import (
    calc_gini,
    calc_gini0,
    calc_modified_gini,
    calc_overdispersion,
    lorenz_baseline_from_pmf,
    lorenz_baseline_from_poisson,
    lorenz_curve,
    simple_lorenz_curve,
)


def test_lorenz_curve_no_weights() -> None:
    df = pd.DataFrame({"count": [10, 20, 30, 40]})
    result = lorenz_curve(df, "count")
    assert isinstance(result, pd.Series)
    assert len(result) == 5
    assert np.isclose(result.loc[0.0], 0.0)
    assert np.isclose(result.loc[0.25], 0.4)
    assert np.isclose(result.loc[0.5], 0.7)
    assert np.isclose(result.loc[0.75], 0.9)
    assert np.isclose(result.loc[1.0], 1.0)


def test_lorenz_curve_with_weights() -> None:
    df = pd.DataFrame({"count": [10, 20, 30, 40], "weight": [1, 4, 3, 2]})
    result = lorenz_curve(df, "count", weight_col="weight")
    assert len(result) == 5
    assert np.isclose(result.loc[0.0], 0.0)
    assert np.isclose(result.loc[0.2], 0.4)
    assert np.isclose(result.loc[0.5], 0.7)
    assert np.isclose(result.loc[0.6], 0.8)
    assert np.isclose(result.loc[1.0], 1.0)


def test_lorenz_curve_degenerate_weights() -> None:
    # this choice of weights is degenerate (ordering becomes arbitrary, result is y=x)
    df = pd.DataFrame({"count": [10, 20, 30, 40], "weight": [1, 2, 3, 4]})
    result = lorenz_curve(df, "count", weight_col="weight")
    assert len(result) == 5
    assert np.isclose(result.loc[0.0], 0.0)
    assert np.isclose(result.loc[0.4], 0.4)
    assert np.isclose(result.loc[0.7], 0.7)
    assert np.isclose(result.loc[0.9], 0.9)
    assert np.isclose(result.loc[1.0], 1.0)


def test_lorenz_curve_no_normalisation() -> None:
    df = pd.DataFrame({"count": [10, 20, 30, 40]})
    result = lorenz_curve(df, "count", normalise_x=False, normalise_y=False)
    assert np.isclose(result.loc[0], 0.0)
    assert np.isclose(result.loc[1], 40.0)
    assert np.isclose(result.loc[2], 70.0)
    assert np.isclose(result.loc[3], 90.0)
    assert np.isclose(result.loc[4], 100.0)
    result = lorenz_curve(df, "count", normalise_y=False)
    assert np.isclose(result.loc[0.0], 0.0)
    assert np.isclose(result.loc[0.25], 40.0)
    assert np.isclose(result.loc[0.5], 70.0)
    assert np.isclose(result.loc[0.75], 90.0)
    assert np.isclose(result.loc[1.0], 100.0)
    result = lorenz_curve(df, "count", normalise_x=False)
    assert np.isclose(result.loc[0], 0.0)
    assert np.isclose(result.loc[1], 0.4)
    assert np.isclose(result.loc[2], 0.7)
    assert np.isclose(result.loc[3], 0.9)
    assert np.isclose(result.loc[4], 1.0)


def test_lorenz_curve_empty_dataframe() -> None:
    df = pd.DataFrame({"count": []})
    with pytest.raises(ValueError):  # noqa: F821
        _ = lorenz_curve(df, "count")


def test_simple_lorenz_curve() -> None:
    counts = pd.Series([10, 20, 30, 40])
    result = simple_lorenz_curve(counts)
    assert isinstance(result, pd.Series)
    assert len(result) == 5
    assert np.isclose(result.loc[0.0], 0.0)
    assert np.isclose(result.loc[1.0], 1.0)


def test_lorenz_baseline_from_poisson_small_lambda() -> None:
    lambda_ = 2.0
    result = lorenz_baseline_from_poisson(lambda_)
    assert isinstance(result, pd.Series)
    assert np.isclose(result.loc[0.0], 0.0)
    assert np.isclose(result.loc[1.0], 1.0)
    assert result.index.is_monotonic_increasing


def test_lorenz_baseline_from_poisson_large_lambda() -> None:
    lambda_ = 10.0
    result = lorenz_baseline_from_poisson(lambda_)
    assert isinstance(result, pd.Series)
    assert np.isclose(result.loc[0.0], 0.0)
    assert np.isclose(result.loc[1.0], 1.0)
    assert result.index.is_monotonic_increasing
    assert len(result.index.unique()) == len(result)  # no duplicates


def test_lorenz_baseline_from_pmf() -> None:
    pmf = pd.Series({0: 0.1, 1: 0.3, 2: 0.4, 3: 0.2})
    result = lorenz_baseline_from_pmf(pmf)
    assert isinstance(result, pd.Series)
    assert np.isclose(result.loc[0.0], 0.0, atol=1e-10) or 0.0 in result.index
    assert np.isclose(result.loc[1.0], 1.0)
    assert result.index.is_monotonic_increasing


def test_calc_gini() -> None:
    lorenz = pd.Series({0.0: 0.0, 0.5: 0.2, 1.0: 1.0})
    gini = calc_gini(lorenz)
    assert isinstance(gini, float)
    assert -2.0 < gini < 1.0


def test_calc_gini_with_ref() -> None:
    lorenz = pd.Series({0.0: 0.0, 0.5: 0.3, 1.0: 1.0})
    ref = pd.Series({0.0: 0.0, 0.5: 0.5, 1.0: 1.0})
    gini = calc_gini(lorenz, ref=ref)
    assert isinstance(gini, float)


def test_calc_gini0() -> None:
    lambda_ = 5.0
    gini0 = calc_gini0(lambda_)
    assert isinstance(gini0, float)
    assert -1.0 < gini0 < 1.0


def test_calc_modified_gini() -> None:
    lorenz = pd.Series({0.0: 0.0, 0.5: 0.2, 1.0: 1.0})
    lambda_ = 5.0
    modified_gini = calc_modified_gini(lorenz, lambda_)
    assert isinstance(modified_gini, float)


def test_calc_overdispersion_poisson() -> None:
    data = pd.Series(np.random.poisson(5, 100))
    overdispersion = calc_overdispersion(data)
    assert isinstance(overdispersion, float)
    assert overdispersion > -1.0


def test_calc_overdispersion_uniform() -> None:
    data = pd.Series([5, 5, 5, 5, 5])
    overdispersion = calc_overdispersion(data)
    assert isinstance(overdispersion, float)
    assert overdispersion < 0.0


if __name__ == "__main__":
    test_lorenz_curve_no_normalisation()
