import numpy as np
import pandas as pd
import pytest

from safer_streets_core.measures import lorenz_curve


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


if __name__ == "__main__":
    test_lorenz_curve_no_normalisation()
