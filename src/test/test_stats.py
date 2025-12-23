import numpy as np
import pandas as pd
import pytest

from safer_streets_core.stats import (
    PoissonGammaModel,
    exponential_fit,
    gamma_fit,
    lognorm_fit,
    nbinom_fit,
    nbinom_poisson_gamma,
    poisson_fit,
    poisson_pvalue,
)


def test_poisson_fit_returns_distribution() -> None:
    data = pd.Series([1, 2, 3, 4, 5])
    result = poisson_fit(data)
    assert result.mean() == data.mean()


def test_nbinom_fit_returns_distribution() -> None:
    data = pd.Series([1, 2, 3, 4, 5])
    result = nbinom_fit(data)
    assert hasattr(result, "pmf")


def test_nbinom_poisson_gamma_returns_distribution() -> None:
    data = pd.Series([1, 2, 3, 4, 5])
    result = nbinom_poisson_gamma(data)
    assert hasattr(result, "pmf")


def test_gamma_fit_returns_distribution() -> None:
    data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gamma_fit(data)
    assert hasattr(result, "pdf")


def test_exponential_fit_returns_distribution() -> None:
    data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = exponential_fit(data)
    assert hasattr(result, "pdf")


def test_lognorm_fit_returns_distribution() -> None:
    data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = lognorm_fit(data)
    assert hasattr(result, "pdf")


def test_poisson_pvalue_returns_float() -> None:
    data = pd.Series([0, 1, 2, 3, 4, 5])
    result = poisson_pvalue(data)
    assert isinstance(result, (float, np.floating))
    assert 0 <= result <= 1


def test_poisson_pvalue_with_custom_mean() -> None:
    data = pd.Series([0, 1, 2, 3, 4, 5])
    result = poisson_pvalue(data, mean=2.5)
    assert isinstance(result, (float, np.floating))


@pytest.fixture
def sample_count_data() -> pd.DataFrame:
    return pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]}, index=["a", "b", "c"])


def test_init(sample_count_data: pd.DataFrame) -> None:
    model = PoissonGammaModel(sample_count_data, random_state=42)
    assert len(model.index) == 3
    assert isinstance(model.rng, np.random.Generator)


def test_means(sample_count_data: pd.DataFrame) -> None:
    model = PoissonGammaModel(sample_count_data, random_state=42)
    means = model.means()
    assert isinstance(means, pd.Series)
    assert len(means) == 3


def test_vars(sample_count_data: pd.DataFrame) -> None:
    model = PoissonGammaModel(sample_count_data, random_state=42)
    vars_result = model.vars()
    assert isinstance(vars_result, pd.Series)
    assert len(vars_result) == 3


def test_resample_lambdas(sample_count_data: pd.DataFrame) -> None:
    model = PoissonGammaModel(sample_count_data, random_state=42)
    model.resample_lambdas()
    lambdas2 = model.lambdas
    assert isinstance(lambdas2, (np.ndarray, pd.Series))


def test_simulate_counts(sample_count_data: pd.DataFrame) -> None:
    model = PoissonGammaModel(sample_count_data, random_state=42)
    sims = model.simulate_counts(n=5)
    assert isinstance(sims, pd.DataFrame)
    assert len(sims) == 3
    assert len(sims.columns) == 5


def test_simulate_counts_with_lambdas(sample_count_data: pd.DataFrame) -> None:
    model = PoissonGammaModel(sample_count_data, random_state=42)
    sims = model.simulate_counts(n=3, return_lambdas=True)
    assert "lambda" in sims.columns
    assert len(sims.columns) == 4


def test_zero_counts_warning() -> None:
    count_data = pd.DataFrame({"col1": [0, 1, 2], "col2": [0, 3, 4]}, index=["x", "y", "z"])
    with pytest.warns(UserWarning, match="Zero counts found"):
        PoissonGammaModel(count_data, random_state=42)


def test_poisson_gamma_model_integration() -> None:
    """Test PoissonGammaModel with realistic data."""
    data = pd.DataFrame({"feature1": [5, 10, 15], "feature2": [3, 8, 12]}, index=["obs1", "obs2", "obs3"])
    model = PoissonGammaModel(data, random_state=42)
    sims = model.simulate_counts(n=10)
    assert sims.shape == (3, 10)


def test_all_distributions_have_pdf_or_pmf() -> None:
    """Verify all distribution types have appropriate methods."""
    data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert hasattr(gamma_fit(data), "pdf")
    assert hasattr(exponential_fit(data), "pdf")
    assert hasattr(lognorm_fit(data), "pdf")


def test_poisson_fit_with_larger_dataset() -> None:
    """Test poisson_fit with larger sample."""
    data = pd.Series(np.random.poisson(3, 1000))
    result = poisson_fit(data)
    assert abs(result.mean() - data.mean()) < 0.5


def test_nbinom_fit_properties() -> None:
    """Verify nbinom_fit produces valid distribution."""
    data = pd.Series([1, 2, 3, 4, 5, 6, 7])
    result = nbinom_fit(data)
    assert result.mean() > 0
