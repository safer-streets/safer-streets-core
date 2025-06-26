from itertools import pairwise, product
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely import LineString
from shapely.geometry import Point, Polygon

import utils
import spatial


def test_format_boundary_as_param() -> None:
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    param = utils.format_boundary_as_param(poly)
    assert isinstance(param, str)
    assert param.count(":") == len(poly.exterior.coords.xy[0]) - 1


def test_tokenize_force_name() -> None:
    assert utils.tokenize_force_name("West Midlands") == "west-midlands"
    assert utils.tokenize_force_name("City of London") == "city-of-london"


def test_monthgen() -> None:
    gen = utils.monthgen(2023, 11)
    assert next(gen) == "2023-11"
    assert next(gen) == "2023-12"
    assert next(gen) == "2024-01"


def test_lorenz_curve_and_gini() -> None:
    data = pd.Series([1, 2, 3, 4, 5])
    lorenz = utils.lorenz_curve(data)
    assert np.isclose(lorenz.iloc[-1], 1.0)
    gini, lorenz2 = utils.calc_gini(data)
    assert 0 <= gini <= 1
    assert isinstance(lorenz2, pd.Series)


def test_spearman_rank_correlation() -> None:
    left = pd.Series([1, 2, 3], index=["a", "b", "c"])
    right = pd.Series([3, 2, 1], index=["a", "b", "c"])
    corr = utils.spearman_rank_correlation(left, right)
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


if __name__ == "__main__":
    test_rank_biased_overlap()



