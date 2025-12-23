import numpy as np
import pandas as pd
import pytest

from safer_streets_core.measures import (
    calc_gini,
    calc_gini0,
    calc_modified_gini,
    calc_overdispersion,
    cosine_similarity,
    diversity_coefficient,
    lorenz_baseline_from_pmf,
    lorenz_baseline_from_poisson,
    lorenz_curve,
    rank_biased_overlap,
    simple_lorenz_curve,
    spearman_rank_correlation,
    spearman_rank_correlation_matrix,
    threshold_f1,
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


def test_spearman_rank_correlation() -> None:
    data = pd.DataFrame(index=["a", "b", "c"], data={"left": [1, 2, 3], "right": [3, 2, 1]})
    corr = spearman_rank_correlation(data)
    assert np.isclose(corr, -1.0)


def test_cosine_similarity() -> None:
    data = pd.DataFrame({"col1": [1, 2, 3], "col2": [1, 2, 3]})
    similarity = cosine_similarity(data)
    assert np.isclose(similarity, 1.0)


def test_cosine_similarity_orthogonal() -> None:
    data = pd.DataFrame({"col1": [1, 0, 0], "col2": [0, 1, 0]})
    similarity = cosine_similarity(data)
    assert np.isclose(similarity, 0.0)


def test_spearman_rank_correlation_matrix() -> None:
    data = pd.DataFrame({"col1": [1, 2, 3], "col2": [3, 2, 1], "col3": [1, 2, 3]})
    corr_matrix = spearman_rank_correlation_matrix(data)
    assert corr_matrix.shape == (3, 3)
    assert np.isclose(corr_matrix[0, 0], 1.0)
    assert np.isclose(corr_matrix[1, 1], 1.0)
    assert np.isclose(corr_matrix[0, 1], -1.0)
    assert np.isclose(corr_matrix[0, 2], 1.0)


def test_spearman_rank_correlation_matrix_empty() -> None:
    data = pd.DataFrame()
    with pytest.raises(AssertionError):
        spearman_rank_correlation_matrix(data)


def test_spearman_rank_correlation_matrix_duplicated_index() -> None:
    data = pd.DataFrame(index=["a", "a", "b"], data={"col1": [1, 2, 3], "col2": [3, 2, 1]})
    with pytest.raises(AssertionError):
        spearman_rank_correlation_matrix(data)


def test_rank_biased_overlap_identical() -> None:
    data = pd.DataFrame({"col1": [1, 2, 3], "col2": [1, 2, 3]})
    rbo = rank_biased_overlap(data)
    assert np.isclose(rbo, 1.0)


def test_rank_biased_overlap_opposite() -> None:
    data = pd.DataFrame({"col1": [1, 2, 3], "col2": [3, 2, 1]})
    rbo = rank_biased_overlap(data)
    assert 0.0 < rbo < 1.0


def test_threshold_f1() -> None:
    data = pd.DataFrame({"col1": [0.1, 0.2, 0.9, 0.8], "col2": [0.15, 0.25, 0.85, 0.75]})
    f1 = threshold_f1(data, threshold=0.5)
    assert isinstance(f1, float)
    assert 0.0 <= f1 <= 1.0


def test_diversity_coefficient_uniform() -> None:
    proportions = pd.Series([0.25, 0.25, 0.25, 0.25])
    div = diversity_coefficient(proportions)
    assert np.isclose(div, 1.0)


def test_diversity_coefficient_single() -> None:
    proportions = pd.Series([1.0])
    div = diversity_coefficient(proportions)
    assert np.isclose(div, 0.0)
