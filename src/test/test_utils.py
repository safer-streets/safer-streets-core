from itertools import pairwise
from pathlib import Path
from typing import get_args

import geopandas as gpd
import pandas as pd
import pytest
from itrx import Itr
from shapely import LineString
from shapely.geometry import Point, Polygon

import safer_streets_core.measures as measures
import safer_streets_core.utils as utils
from safer_streets_core import spatial


def test_format_boundary_as_param() -> None:
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    param = utils.format_boundary_as_param(poly)
    assert isinstance(param, str)
    assert param.count(":") == len(poly.exterior.coords.xy[0]) - 1


def test_tokenize_force_name() -> None:
    # these are the names used in the data.police.uk archives
    tokenized = {
        "hampshire",
        "dyfed-powys",
        "lancashire",
        "suffolk",
        "thames-valley",
        "leicestershire",
        "lincolnshire",
        "greater-manchester",
        "sussex",
        "cambridgeshire",
        "nottinghamshire",
        "humberside",
        "surrey",
        "northamptonshire",
        "north-wales",
        "cumbria",
        "west-midlands",
        "cheshire",
        "kent",
        "hertfordshire",
        "dorset",
        "durham",
        "btp",
        "avon-and-somerset",
        "cleveland",
        "staffordshire",
        "west-mercia",
        "essex",
        "north-yorkshire",
        "wiltshire",
        "gloucestershire",
        "south-wales",
        "gwent",
        "northumbria",
        "south-yorkshire",
        "devon-and-cornwall",
        "northern-ireland",
        "city-of-london",
        "derbyshire",
        "norfolk",
        "bedfordshire",
        "merseyside",
        "metropolitan",
        "west-yorkshire",
        "warwickshire",
    }

    for force in get_args(utils.Force):
        assert utils.tokenize_force_name(force) in tokenized


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
        spatial.get_square_grid(spatial.get_force_boundary("Test"), size=1.0, offset=(2.0, 2.0))  # ty: ignore[invalid-argument-type]


def test_snap_to_street_segment() -> None:
    # Create two points and two segments
    points = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:27700")
    segments = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 1)]), LineString([(1, 1), (1, 2)])], crs="EPSG:27700")
    points = spatial.snap_to_street_segment(points, segments)
    assert "street_segment" in points.columns
    assert "distance" in points.columns


def test_rank_biased_overlap() -> None:
    df = pd.DataFrame(columns=list("abcd"), data=[[0, 0, 0, 1], [0, 1, 2, 2]]).T

    # Steps (ranked in descending order):
    # Iter Left    Right   Union     Inter      Score  Weight(p=0.5)
    # 0    {d}     {c,d}   {c,d}     {d}        1/2    1
    # 1    {c,a,b} {b}     {a,b,c,d} {c,b,d}    3/4    1/2
    # 2    {}      {a}     {a,b,c,d} {c,a,b,d}  1      1/4

    assert measures.rank_biased_overlap(df, 1.0) == pytest.approx(3 / 4)
    assert measures.rank_biased_overlap(df, 0.5) == pytest.approx(9 / 14)


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


def test_month_subtraction() -> None:
    m = utils.Month(2023, 2)
    m2 = m - 1
    assert m2.year == 2023
    assert m2.month == 1
    m3 = m - 2
    assert m3.year == 2022
    assert m3.month == 12
    m4 = m - 20
    assert m4.year == 2021
    assert m4.month == 6


def test_month_comparison() -> None:
    m1 = utils.Month(2023, 5)
    m2 = utils.Month(2023, 6)
    m3 = utils.Month(2024, 1)
    assert m1 < m2
    assert m2 < m3
    assert not (m3 < m1)


def test_monthgen_basic_iteration() -> None:
    start = utils.Month(2023, 5)
    end = utils.Month(2023, 8)
    mr = utils.monthgen(start, end=end)
    months = list(mr)
    assert len(months) == 3
    assert months[0].year == 2023 and months[0].month == 5
    assert months[1].year == 2023 and months[1].month == 6
    assert months[2].year == 2023 and months[2].month == 7


def test_monthgen_no_end() -> None:
    start = utils.Month(2023, 1)
    months = Itr(utils.monthgen(start))
    assert all(m.month == (i % 12) + 1 and m.year == 2023 + i // 12 for i, m in months.take(5).enumerate())


def test_monthgen_end_equal_start() -> None:
    start = utils.Month(2023, 5)
    mr = utils.monthgen(start, end=start)
    with pytest.raises(StopIteration):
        next(mr)


def test_monthgen_end_before_start() -> None:
    start = utils.Month(2023, 5)
    end = utils.Month(2023, 3)
    mr = Itr(utils.monthgen(start, end=end))
    assert next(mr) == start
    assert next(mr) == start - 1
    with pytest.raises(StopIteration):
        next(mr)


def test_monthgen_pairwise() -> None:
    start = utils.Month(2020, 12)
    end = utils.Month(2022, 5)
    m = Itr(utils.monthgen(start, end=end))
    pairs = m.pairwise().collect()
    assert pairs[0][0] == start
    assert pairs[-1][-1] + 1 == end

    for p0, p1 in pairwise(pairs):
        assert p0[1] == p1[0]


def test_fix_force_name() -> None:
    assert utils.fix_force_name("Metropolitan") == "Metropolitan Police"
    assert utils.fix_force_name("Devon and Cornwall") == "Devon & Cornwall"
    assert utils.fix_force_name("City of London") == "London, City of"
    assert utils.fix_force_name("Dyfed Powys") == "Dyfed-Powys"
    # unmapped names pass through unchanged
    assert utils.fix_force_name("West Yorkshire") == "West Yorkshire"


def test_archive_path(monkeypatch) -> None:
    monkeypatch.setattr(utils, "data_dir", lambda: Path("/data"))
    assert utils.archive_path() == Path("/data/police_uk_crime_data_latest.zip")
    assert utils.archive_path("2024-01") == Path("/data/police_uk_crime_data_2024-01.zip")


def test_extracted_data_path(monkeypatch) -> None:
    monkeypatch.setattr(utils, "data_dir", lambda: Path("/data"))
    month = utils.Month(2024, 3)
    assert utils.extracted_data_path(month, "west-yorkshire") == Path(
        "/data/extracted/2024-03-west-yorkshire-street.parquet"
    )


def test_month_parse_str() -> None:
    m = utils.Month.parse_str("2024-07")
    assert m.year == 2024
    assert m.month == 7


def test_month_eq_with_non_month() -> None:
    assert (utils.Month(2024, 1) == "not a month") is False


def test_month_hashable_and_usable_in_set() -> None:
    months = {utils.Month(2024, 1), utils.Month(2024, 1), utils.Month(2024, 2)}
    assert len(months) == 2


def test_month_ge() -> None:
    assert utils.Month(2024, 2) >= utils.Month(2024, 1)
    assert utils.Month(2024, 1) >= utils.Month(2024, 1)
    assert not (utils.Month(2024, 1) >= utils.Month(2024, 2))


def test_x_interp() -> None:
    data = pd.Series(index=[0.0, 1.0, 2.0], data=[0.0, 10.0, 20.0])
    result = utils.x_interp(data, pd.Index([0.5, 1.5]))
    assert result.loc[0.5] == pytest.approx(5.0)
    assert result.loc[1.5] == pytest.approx(15.0)


def test_y_interp() -> None:
    data = pd.Series(index=[0.0, 1.0, 2.0], data=[0.0, 10.0, 20.0])
    result = utils.y_interp(data, pd.Index([5.0, 15.0]))
    assert result.loc[5.0] == pytest.approx(0.5)
    assert result.loc[15.0] == pytest.approx(1.5)


def test_get_crime_counts() -> None:
    crimes = pd.DataFrame({"spatial_unit": ["a", "a", "b"]})
    features = gpd.GeoDataFrame(index=pd.Index(["a", "b", "c"], name="spatial_unit"))
    counts = utils.get_crime_counts(crimes, features)
    assert counts.loc["a"] == 2
    assert counts.loc["b"] == 1
    # feature with no crimes is still present, with a zero count
    assert counts.loc["c"] == 0


def test_get_monthly_crime_counts() -> None:
    crimes = pd.DataFrame(
        {
            "spatial_unit": ["a", "a", "b"],
            "Month": ["2024-01", "2024-02", "2024-01"],
        }
    )
    features = gpd.GeoDataFrame(index=pd.Index(["a", "b", "c"], name="spatial_unit"))
    counts = utils.get_monthly_crime_counts(crimes, features)
    assert counts.loc["a", "2024-01"] == 1
    assert counts.loc["a", "2024-02"] == 1
    assert counts.loc["b", "2024-01"] == 1
    # feature with no crimes is present and zeroed across all months
    assert (counts.loc["c"] == 0).all()


def test_random_crime_data_by_feature_unweighted() -> None:
    features = gpd.GeoDataFrame(
        index=pd.Index(["a", "b", "c"], name="spatial_unit"),
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
    )
    months = [utils.Month(2024, 1), utils.Month(2024, 2)]
    result = utils.random_crime_data_by_feature(50, features, months, random_state=42)
    assert len(result) == 50
    assert set(result.columns) == {"Month", "spatial_unit", "Crime type"}
    assert result.spatial_unit.isin(["a", "b", "c"]).all()
    assert (result["Crime type"] == "Random").all()


def test_random_crime_data_by_feature_weighted_requires_weight_column() -> None:
    features = gpd.GeoDataFrame(
        index=pd.Index(["a", "b"], name="spatial_unit"),
        geometry=[Point(0, 0), Point(1, 1)],
    )
    with pytest.raises(AssertionError):
        utils.random_crime_data_by_feature(5, features, [utils.Month(2024, 1)], weighted=True)


def test_random_crime_data_by_feature_weighted() -> None:
    features = gpd.GeoDataFrame(
        index=pd.Index(["a", "b"], name="spatial_unit"),
        geometry=[Point(0, 0), Point(1, 1)],
        data={"weight": [0.0, 1.0]},
    )
    result = utils.random_crime_data_by_feature(20, features, [utils.Month(2024, 1)], weighted=True, random_state=1)
    # all weight is on "b", so no sample should ever land on "a"
    assert (result.spatial_unit == "b").all()


def test_random_crime_data_by_point() -> None:
    boundary = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])],
        crs="EPSG:27700",
    )
    result = utils.random_crime_data_by_point(15, boundary, [utils.Month(2024, 1)], random_state=7)
    assert len(result) == 15
    assert (result["Crime type"] == "Random").all()
    assert result.crs == boundary.crs


def test_data_source(monkeypatch, tmp_path) -> None:
    (tmp_path / "data_sources.json").write_text('{"foo": "https://example.com/foo.zip"}')
    monkeypatch.setattr(utils, "data_dir", lambda: tmp_path)
    assert utils.data_source("foo") == "https://example.com/foo.zip"
    with pytest.raises(ValueError, match="does not map to a data source"):
        utils.data_source("missing")


if __name__ == "__main__":
    test_monthgen_pairwise()
