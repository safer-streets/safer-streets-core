import os
import warnings
from calendar import monthrange
from collections.abc import Iterable, Iterator
from functools import cache
from pathlib import Path
from typing import Any, Literal, Self
from warnings import deprecated
from zipfile import ZipFile

import geopandas as gpd
import humanleague as hl
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from itrx import Itr
from scipy.stats import poisson
from shapely import Polygon

from safer_streets_core.models import Neighbourhoods, RawPolygon

CATEGORIES = ("Violence and sexual offences", "Anti-social behaviour", "Possession of weapons")

Force = Literal[
    "Avon and Somerset",
    "Bedfordshire",
    "BTP",
    "Cambridgeshire",
    "Cheshire",
    "City of London",
    "Cleveland",
    "Cumbria",
    "Derbyshire",
    "Devon and Cornwall",
    "Dorset",
    "Durham",
    "Dyfed Powys",
    "Essex",
    "Gloucestershire",
    "Greater Manchester",
    "Gwent",
    "Hampshire",
    "Hertfordshire",
    "Humberside",
    "Kent",
    "Lancashire",
    "Leicestershire",
    "Lincolnshire",
    "Merseyside",
    "Metropolitan",
    "Norfolk",
    "North Wales",
    "North Yorkshire",
    "Northamptonshire",
    "Northern Ireland",
    "Northumbria",
    "Nottinghamshire",
    "South Wales",
    "South Yorkshire",
    "Staffordshire",
    "Suffolk",
    "Surrey",
    "Sussex",
    "Thames Valley",
    "Warwickshire",
    "West Mercia",
    "West Midlands",
    "West Yorkshire",
    "Wiltshire",
]


@cache
def data_dir() -> Path:
    """Returns the data folder path or raises an error if not set"""
    load_dotenv()
    return Path(os.environ["SAFER_STREETS_DATA_DIR"])


def format_boundary_as_param(polygon: Polygon) -> str:
    xy = zip(*(c.tolist() for c in polygon.exterior.coords.xy), strict=False)
    return ":".join(f"{x:.3f},{y:.3f}" for x, y in xy)


def _get_boundary(force: Force, neighbourhood_id: str) -> Polygon:
    try:
        return RawPolygon(
            requests.get(f"{POLICE_API_BASE_URL}/{force}/{neighbourhood_id}/boundary").json()
        ).to_shapely()
    except Exception:
        return Polygon()


@cache
def get_raw_geog_lookup() -> pd.DataFrame:
    # https://www.arcgis.com/sharing/rest/content/items/80592949bebd4390b2cbe29159a75ef4/data
    return pd.read_csv(data_dir() / "PCD_OA21_LSOA21_MSOA21_LAD_FEB25_UK_LU.zip")


def get_geog_lookup(geog_from: str, geogs_to: list[str]) -> pd.DataFrame:
    lookup = get_raw_geog_lookup()[[geog_from, *geogs_to]].drop_duplicates()
    return lookup.set_index(geog_from)


POLICE_API_BASE_URL = "http://data.police.uk/api"
POLICE_DATA_BASE_URL = "http://data.police.uk/data"
ARCHIVE_TEMPLATE = f"{data_dir()}/police_uk_crime_data_{{}}.zip"
DATA_TEMPLATE = f"{data_dir()}/extracted/{{month}}-{{force}}-street.parquet"


class Month:
    def __init__(self, y: int, m: int) -> None:
        assert 1900 <= y <= 2100
        assert 1 <= m <= 12
        self.y = y
        self.m = m - 1  # internally use 0-11

    @property
    def year(self) -> int:
        return self.y

    @property
    def month(self) -> int:
        return self.m + 1

    @property
    def days(self) -> int:
        "Returns no of days in month"
        return monthrange(self.y, self.m + 1)[1]

    @staticmethod
    def parse_str(yyyy_mm: str) -> "Month":
        return Month(*(int(n) for n in yyyy_mm.split("-")))

    def __add__(self, months: int) -> "Month":
        if months < 0:
            return self - abs(months)
        m = self.m + months
        years = m // 12
        m = m % 12
        y = self.y + years
        return Month(y, m + 1)

    def __sub__(self, months: int) -> "Month":
        if months < 0:
            return self + abs(months)
        m = self.m - months
        years = m // 12  # this will be negative if m < 0
        m = m % 12
        y = self.y + years
        return Month(y, m + 1)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.y == other.y and self.m == other.m

    def __lt__(self, other: Self) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.y < other.y or (self.y == other.y and self.m < other.m)

    def __ge__(self, other: Self) -> bool:
        return not self < other

    def __repr__(self) -> str:
        return f"{self.y}-{self.m + 1:02}"

    def __hash__(self) -> int:
        return hash(self.y * 12 + self.m)


def monthgen(start: Month, end: Month | None = None, backwards: bool = False) -> Iterator[Month]:
    """
    Generates months starting from `start` until `end`.
    If `end` is None, it generates indefinitely.
    """

    if end:
        backwards = end < start

    current = start
    while current != end:
        yield current
        current = current - 1 if backwards else current + 1


def tokenize_force_name(force_name: Force) -> str:
    """Tokenize the force name for use in file paths."""
    # TODO fix this to handle cases like "Metropolitan Police" vs "metropolitan"
    return force_name.replace(" ", "-").lower()


def get_neighbourhood_boundaries(force: Force) -> gpd.GeoDataFrame:
    neighbourhoods = Neighbourhoods(
        requests.get(f"{POLICE_API_BASE_URL}/{tokenize_force_name(force)}/neighbourhoods").json()
    )
    neighbourhood_boundaries = gpd.GeoDataFrame(
        index=tuple(n.id for n in neighbourhoods),
        data={"name": (n.name for n in neighbourhoods)},
        geometry=[_get_boundary(force, n.id) for n in neighbourhoods],
        crs="epsg:4326",
    ).to_crs(epsg=27700)
    return neighbourhood_boundaries


def _format_crime_data(crime_data: pd.DataFrame, keep_lonlat: bool, filters: dict[str, Any]) -> gpd.GeoDataFrame:
    # drop crimes with no location
    crime_data = crime_data.dropna(subset=["Longitude", "Latitude"])

    # apply any filters
    for column, value in (filters or {}).items():
        crime_data = crime_data[crime_data[column] == value]

    return gpd.GeoDataFrame(
        crime_data.rename(columns={"Longitude": "lon", "Latitude": "lat"})
        if keep_lonlat
        else crime_data.drop(columns=["Longitude", "Latitude"]),
        geometry=gpd.points_from_xy(crime_data.Longitude, crime_data.Latitude),
        crs="EPSG:4326",
    ).to_crs(epsg=27700)


@deprecated("Use load_crime_data (plus the extract script if necessary)")
def extract_crime_data(
    force: Force, *, keep_lonlat: bool = False, filters: dict[str, Any] | None = None
) -> gpd.GeoDataFrame:
    """
    Extracts crime data for a given force from the latest archive.
    Use keep_lonlat for rendering  streamlit maps
    """
    archive = Path(ARCHIVE_TEMPLATE.format("latest"))

    if not archive.exists():
        download_archive("latest")

    force_identifier = tokenize_force_name(force)

    with ZipFile(archive) as bulk_data:
        crimes = []
        for file in bulk_data.namelist():
            if f"{force_identifier}-street" in file:
                crimes.append(pd.read_csv(bulk_data.open(file)))
        crime_data = pd.concat(crimes).set_index("Crime ID").drop(columns=["Last outcome category", "Context"])

    return _format_crime_data(crime_data, keep_lonlat, filters or {})


def random_crime_data_by_point(
    n: int, boundary: gpd.GeoDataFrame, months: list, *, seed: int = 19937
) -> gpd.GeoDataFrame:
    """Sample within boundary. Larger features will tend to get more crimes"""
    rng = np.random.default_rng(seed)
    random = gpd.GeoDataFrame(
        geometry=boundary.sample_points(n, rng=rng).explode().to_list(),
        data={"Month": rng.choice(months, n), "Crime type": "Random"},
        crs=boundary.crs,
    )
    return random


def quasirandom_crime_data_by_point(N: int, boundary: gpd.GeoDataFrame, months: list[Any]) -> gpd.GeoDataFrame:
    """Uses Sobol sequence to quasirandomly sample (x, y, t)"""
    minx, miny, maxx, maxy = boundary.bounds.iloc[0]
    deltax = maxx - minx
    deltay = maxy - miny
    area = deltax * deltay

    # oversample to get approximately N within boundary
    Nadj = int(N * area / boundary.area.sum() + 0.5)
    M = len(months)

    def split(x, y, t) -> tuple[tuple[float, float], Any]:
        return (x, y), months[int(t * M)]

    point_seq, month_seq = Itr(hl.SobolSequence(3)).take(Nadj).starmap(split).unzip()
    x, y = point_seq.unzip()

    quasirandom = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(
            np.array(x.collect(list)) * deltax + minx, np.array(y.collect(list)) * deltay + miny
        ),
        data={"Month": month_seq.collect(list), "Crime type": "Quasirandom"},
        crs=boundary.crs,
    )
    return quasirandom.sjoin(boundary[["geometry"]]).drop(columns=["index_right"], errors="ignore")


def random_crime_data_by_feature(
    n: int, features: gpd.GeoDataFrame, months: list, *, weighted: bool = False, seed: int = 19937
) -> pd.DataFrame:
    """Sample features. Larger features won't tend to get more crimes"""
    rng = np.random.default_rng(seed)

    extra_args = {}
    if weighted:
        assert "weight" in features.columns, "features must have a 'weight' columns when weights=True"
        extra_args["p"] = features.weight / features.weight.sum()

    random = pd.DataFrame(
        data={
            "Month": rng.choice(months, n),
            "spatial_unit": rng.choice(features.index, n, **extra_args),
            "Crime type": "Random",
        }
    )

    return random


def download_archive(name: str = "latest") -> bool:
    """
    Downloads the latest police data archive and saves it to CRIME_ARCHIVE.
    To force a download, delete the existing CRIME_ARCHIVE file, or just explicitly call this function.
    """
    url = f"{POLICE_DATA_BASE_URL}/archive/{name}.zip"
    MB = 1024 * 1024

    local_file = Path(ARCHIVE_TEMPLATE.format(name))
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        size = int(response.headers.get("content-length", 0))
        print(f"Downloading archive to {local_file} ({size // MB}MB)", end="")

        with open(local_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=MB):
                if chunk:
                    f.write(chunk)
                    print(".", end="", flush=True)
        print("\nDownload complete.")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")
    except OSError as e:
        print(f"Error writing file {local_file}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return False


def load_crime_data(
    force: Force, months: Iterable[Month], *, keep_lonlat: bool = False, filters: dict[str, Any] | None = None
) -> gpd.GeoDataFrame:
    data = []
    force_identifier = tokenize_force_name(force)
    for month in months:
        path = Path(DATA_TEMPLATE.format(month=month, force=force_identifier))
        if path.is_file():
            data.append(pd.read_parquet(path))
        else:
            warnings.warn(f"Crime data not found: {path}", stacklevel=2)

    return _format_crime_data(pd.concat(data), keep_lonlat, filters or {})


def get_crime_counts(crimes: pd.DataFrame, features: gpd.GeoDataFrame) -> pd.DataFrame:
    "Group by spatial unit and count, ensuring features with no crimes are accounted for"
    return (
        crimes.groupby("spatial_unit")
        .apply(len, include_groups=False)
        .rename("count")
        .reindex(features.index, fill_value=0)
    )


def get_monthly_crime_counts(crimes: pd.DataFrame, features: gpd.GeoDataFrame) -> pd.DataFrame:
    "Group by spatial unit/month and count, ensuring features with no crimes are accounted for"
    return (
        crimes.groupby(["Month", "spatial_unit"])
        .apply(len, include_groups=False)
        .unstack(level="Month", fill_value=0)
        .reindex(features.index, fill_value=0)
        .sort_index()
    ).reindex(features.index, fill_value=0)


def latest_month() -> Month:
    "Returns the most recent month found in the (extracted) archives"
    files = (data_dir() / "extracted").glob("*-street.parquet")
    return Itr(files).map(lambda file: Month.parse_str(file.name[:7])).max()


def x_interp(data: pd.Series, new_x: pd.Index) -> pd.Series:
    """Return linearly interpolated y values at new_x from data"""
    combined_index = data.index.union(new_x)  # .drop_duplicates()
    return data.reindex(combined_index).interpolate(method="linear").loc[new_x].rename("y")


def y_interp(data: pd.Series, new_y: pd.Index) -> pd.Series:
    """Return linearly interpolated x values for new_y (inverts data)"""
    data_inv = pd.Series(index=data, data=data.index)
    # can have duplicated y values in data, but when inverted index must be unique
    data_inv = data_inv[~data_inv.index.duplicated()]
    combined_index = data_inv.index.union(new_y, sort=True)  # .drop_duplicates()
    return data_inv.reindex(combined_index).interpolate(method="linear").loc[new_y].rename("x")


# TODO move or replace all of below in measures.py


@deprecated("Use measures.simple_lorenz_curve")
def lorenz_curve(data: pd.Series, *, percentiles: bool = False) -> pd.Series:
    full = data.sort_values().cumsum() / data.sum()
    if percentiles:
        x = np.linspace(0, 1, 101)
        return pd.Series(index=1 - x, data=1 - np.percentile(full, x * 100)).sort_index()
    # normalise the x axis
    return (1 - full.set_axis(1 - np.linspace(0, 1, len(full)))).sort_index()


@deprecated("Use measures.lorenz_curve")
def weighted_lorenz_curve(
    data: pd.DataFrame, *, data_col: str, weight_col: str, percentiles: bool = False
) -> pd.Series:
    tempdf = data[[data_col, weight_col]].copy()
    tempdf["ordering"] = tempdf[data_col] / tempdf[weight_col]
    tempdf = tempdf.sort_values(by="ordering")
    full = tempdf[data_col].cumsum() / tempdf[data_col].sum()
    index = tempdf[weight_col].cumsum() / tempdf[weight_col].sum()

    if percentiles:
        raise NotImplementedError("TODO if required - interpolate (index, full)")
    return (1 - full.set_axis(1 - index)).sort_index()


@deprecated("Use measures.lorenz_baseline_from_poisson")
def poisson_lorenz_curve(lambda_: float, percentiles: bool = False) -> pd.Series:
    dist = poisson(lambda_)
    length = 5
    threshold = 1.0 - np.finfo(float).eps
    while dist.cdf(length) < threshold:
        length += 1
    cdf = [dist.cdf(k) for k in range(-1, length + 1)]
    # flip curve
    l0 = pd.Series(index=[1 - x for x in (0, *cdf[1:])], data=[1 - y for y in (0, *cdf[:-1])]).sort_index()
    l0[1.0] = 1.0  # this overwrites rather than appends so avoids duplicates
    if percentiles:
        x = np.linspace(0, 1, 101)
        return pd.Series(index=x, data=np.interp(x, l0.index, l0))
    return l0


@deprecated("Use measures.calc_gini")
def calc_gini(data: pd.Series) -> tuple[float, pd.Series]:
    lorenz = lorenz_curve(data, percentiles=True)
    # trapezoidal rule (flipped (area above) scaled by 100 - x axis is %)
    gini = 1.0 - (1.0 - lorenz).rolling(2).mean().sum() / 50.0
    return gini, lorenz


@deprecated("Use measures.lorenz_baseline_from_poisson")
def poisson_lorenz_area(lambda_: float) -> float:
    # work out how many terms we need to sum
    dist = poisson(lambda_)
    length = 5
    threshold = 1 - np.finfo(float).eps
    while dist.cdf(length) < threshold:
        length += 1

    # sum( p(k) * (P(k-1) + P(k-2)) / 2)
    pdf = dist.pmf(range(1, length))
    cdf = (dist.cdf(range(-1, length - 2)) + dist.cdf(range(length - 1))) / 2
    return pdf @ cdf


@deprecated("Do not use")
def calc_adjusted_gini(lorenz: pd.Series, lambda_: float) -> float:
    """
    Calculate the adjusted Gini coefficient from a Lorenz curve and the Poisson intensity.
    """

    A0 = poisson_lorenz_area(lambda_)

    A = (1 - lorenz).sum() / len(lorenz)
    return (A0 - A) / A0
