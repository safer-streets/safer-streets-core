from itertools import pairwise

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely import LineString
from shapely.geometry import Point, Polygon

import spatial
import utils


def test_format_boundary_as_param() -> None:
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    param = utils.format_boundary_as_param(poly)
    assert isinstance(param, str)
    assert param.count(":") == len(poly.exterior.coords.xy[0]) - 1


def test_tokenize_force_name() -> None:
    assert utils.tokenize_force_name("West Midlands") == "west-midlands"
    assert utils.tokenize_force_name("City of London") == "city-of-london"


def test_lorenz_curve_and_gini() -> None:
    data = pd.Series([1, 2, 3, 4, 5])
    lorenz = utils.lorenz_curve(data)
    assert np.isclose(lorenz.iloc[-1], 1.0)
    gini, lorenz2 = utils.calc_gini(data)
    assert 0 <= gini <= 1
    assert isinstance(lorenz2, pd.Series)


def test_spearman_rank_correlation() -> None:
    data = pd.DataFrame(index=["a", "b", "c"], data={"left": [1, 2, 3], "right": [3, 2, 1]})
    corr = utils.spearman_rank_correlation(data)
    assert np.isclose(corr, -1.0)


def test_get_square_grid_offset_assertion(monkeypatch) -> None:
    # Patch get_force_boundary to return a simple square
    monkeypatch.setattr(
        spatial,
        "get_force_boundary",
        lambda name: gpd.GeoDataFrame(
            geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])],
            crs="EPSG:27700",
            data={"PFA23CD": ["X"], "PFA23NM": ["Test"]},
        ),
    )
    with pytest.raises(AssertionError):
        spatial.get_square_grid(spatial.get_force_boundary("Test"), size=1.0, offset=(2.0, 2.0))


def test_snap_to_street_segment() -> None:
    # Create two points and two segments
    points = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:27700")
    segments = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 1)]), LineString([(1, 1), (1, 2)])], crs="EPSG:27700")
    points = spatial.snap_to_street_segment(points, segments)
    assert "street_segment" in points.columns
    assert "distance" in points.columns


def test_rank_biased_overlap() -> None:
    df = pd.DataFrame(columns=list("abcd"), data=[[0, 0, 0, 1], [0, 1, 2, 2]]).T

    # Steps:
    # Iter Left     Right   Union     Inter     Score  Weight(p=0.5)
    # 0    {a,b,c}  {a}     {a,b,c}   {a}       1/3    1
    # 1    {d}      {b}     {a,b,c,d} {a,b}     1/2    1/2
    # 2             {c,d}   {a,b,c,d} {a,b,c,d} 1      1/4
    assert utils.rank_biased_overlap(df, 1.0) == pytest.approx(11 / 18)
    assert utils.rank_biased_overlap(df, 0.5) == pytest.approx(5 / 6 / (1 + 0.5 + 0.25))


def test_month_init_and_properties() -> None:
    m = utils.Month(2023, 5)
    assert m.year == 2023
    assert m.month == 5
    assert repr(m) == "2023-05"


def test_month_invalid_year() -> None:
    with pytest.raises(AssertionError):
        utils.Month(1800, 5)
    with pytest.raises(AssertionError):
        utils.Month(2200, 5)


def test_month_invalid_month() -> None:
    with pytest.raises(AssertionError):
        utils.Month(2023, 0)
    with pytest.raises(AssertionError):
        utils.Month(2023, 13)


def test_month_addition() -> None:
    m = utils.Month(2023, 11)
    m2 = m + 1
    assert m2.year == 2023
    assert m2.month == 12
    m3 = m + 2
    print(m3)
    assert m3.year == 2024
    assert m3.month == 1
    m4 = m + 20
    assert m4.year == 2025
    assert m4.month == 7


def test_month_comparison() -> None:
    m1 = utils.Month(2023, 5)
    m2 = utils.Month(2023, 6)
    m3 = utils.Month(2024, 1)
    assert m1 < m2
    assert m2 < m3
    assert not (m3 < m1)


def test_monthrange_basic_iteration() -> None:
    start = utils.Month(2023, 5)
    end = utils.Month(2023, 8)
    mr = utils.MonthRange(start, end=end)
    months = list(mr)
    assert len(months) == 3
    assert months[0].year == 2023 and months[0].month == 5
    assert months[1].year == 2023 and months[1].month == 6
    assert months[2].year == 2023 and months[2].month == 7


def test_monthrange_no_end() -> None:
    start = utils.Month(2023, 1)
    mr = utils.MonthRange(start)
    # Should be an infinite iterator, so just take first 5
    months = [next(mr) for _ in range(50)]
    assert all(m.month == (i % 12) + 1 and m.year == 2023 + i // 12 for i, m in enumerate(months))


def test_monthrange_end_equal_start() -> None:
    start = utils.Month(2023, 5)
    mr = utils.MonthRange(start, end=start)
    with pytest.raises(StopIteration):
        next(mr)


def test_monthrange_end_before_start() -> None:
    start = utils.Month(2023, 5)
    end = utils.Month(2023, 4)
    mr = utils.MonthRange(start, end=end)
    with pytest.raises(StopIteration):
        next(mr)


def test_monthrange_pairwise() -> None:
    start = utils.Month(2020, 12)
    end = utils.Month(2022, 5)
    m = utils.MonthRange(start, end=end)
    pairs = list(pairwise(m))
    assert pairs[0][0] == start
    assert pairs[-1][-1] + 1 == end

    for p0, p1 in pairwise(pairs):
        assert p0[1] == p1[0]


if __name__ == "__main__":
    test_monthrange_pairwise()
