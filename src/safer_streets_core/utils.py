import os
import warnings
from calendar import monthrange
from collections.abc import Iterable, Iterator
from functools import cache
from itertools import zip_longest
from pathlib import Path
from typing import Any, Literal, Self
from warnings import deprecated
from zipfile import ZipFile

import geopandas as gpd
import humanleague as hl
import numpy as np
import numpy.typing as npt
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
    return crimes.groupby("spatial_unit")["Crime type"].count().rename("count").reindex(features.index, fill_value=0)


def get_monthly_crime_counts(crimes: pd.DataFrame, features: gpd.GeoDataFrame) -> pd.DataFrame:
    "Group by spatial unit/month and count, ensuring features with no crimes are accounted for"
    return (
        crimes.groupby(["Month", "spatial_unit"])["Crime type"]
        .count()
        .unstack(level="Month", fill_value=0)
        .reindex(features.index, fill_value=0)
        .sort_index()
    ).reindex(features.index, fill_value=0)


def latest_month() -> Month:
    "Returns the most recent month found in the (extracted) archives"
    files = (data_dir() / "extracted").glob("*-street.parquet")
    return Itr(files).map(lambda file: Month.parse_str(file.name[:7])).max()


# def extract_monthly_crime_data(
#     force: Force, month: Month, *, keep_lonlat: bool = False, filters: dict[str, Any] | None = None
# ) -> pd.DataFrame:
#     # NB data.police.uk says:
#  # > "With the exception of the latest month’s archive, the data on this page is out of date and should not be used."
#     # newest archive to contain oldest data is Apr 2017, which has data from Dec 2010
#     # for dates after Apr 2017, get the archive 2y 11months after required date, extract the files for the first month
#     # in the data (this will be the newest archive)

#     # filters allows basic filtering on values in specific columns, e.g. {"Crime type": "Anti-social behaviour"}

#     if month < Month(2010, 12):
#         raise ValueError(f"Data for {month} is not available")
#     elif Month(2022, 5) < month:  # TODO this will need regular updating
#         archive = "latest"
#     elif month < Month(2014, 6):
#         archive = "2017-04"
#     else:
#         archive = str(list(MonthRange(month, end=month + 36))[-1])

#     local_file = ARCHIVE_TEMPLATE.format(archive)
#     if not Path(local_file).exists():
#         download_archive(archive)

#     crime_data = pd.DataFrame()
#     with ZipFile(local_file) as bulk_data:
#         for file in bulk_data.namelist():
#             if f"{month}-{tokenize_force_name(force)}-street" in file:
#                 crime_data = (
#                     pd.read_csv(bulk_data.open(file))
#                     .set_index("Crime ID")
#                     .drop(columns=["Last outcome category", "Context"])
#                 )
#                 break
#     if crime_data.empty:
#         raise ValueError(f"No crime data available for {force} in {month}")

#     return _format_crime_data(crime_data, keep_lonlat, filters or {})

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


def calc_adjusted_gini(lorenz: pd.Series, lambda_: float) -> float:
    """
    Calculate the adjusted Gini coefficient from a Lorenz curve and the Poisson intensity.
    """

    A0 = poisson_lorenz_area(lambda_)

    A = (1 - lorenz).sum() / len(lorenz)
    return (A0 - A) / A0


def _spearman_rank_correlation_impl(diff: pd.Series) -> float:
    n = len(diff)
    return 1 - 6 * (diff**2).sum() / (n * (n * n - 1))


def spearman_rank_correlation(ranks: pd.DataFrame) -> float:
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


# # based on code from https://towardsdatascience.com/rbo-v-s-kendall-tau-to-compare-ranked-lists-of-items-8776c5182899/
# def rank_biased_overlap1(left: list[Any], right: list[Any], p: float = 0.9) -> float:
#     k = max(len(left), len(right))
#     x_k = len(set(left).intersection(right))
#     summation = sum(p**i * len(set(left[:i]).intersection(right[:i])) / i for i in range(1, k + 1))
#     return (float(x_k) / k * p**k) + ((1 - p) / p * summation)


# def rbo_weight(p: float, n: int) -> float:
#     """
#     Contribution of top n rankings for a given decay constant p
#     e.g. rbo_weight(0.9, 10) ~= 0.856
#     """
#     assert 0 < p <= 1
#     s = sum(p**i / i for i in range(1, n))
#     return 1.0 - p ** (n - 1) + ((1 - p) / p * n * (np.log(1 / (1 - p)) - s))


# def rank_biased_overlap2(l1, l2, p=0.9):
#     """
#     Calculates Ranked Biased Overlap (RBO) score.
#     l1 -- Ranked List 1
#     l2 -- Ranked List 2
#     """
#     l1 = l1 or []
#     l2 = l2 or []

#     sl, ll = sorted([(len(l1), l1), (len(l2), l2)])
#     s, S = sl
#     l, L = ll
#     if s == 0:
#         return 0

#     # Calculate the overlaps at ranks 1 through l
#     # (the longer of the two lists)
#     ss = set([])  # contains elements from the smaller list till depth i
#     ls = set([])  # contains elements from the longer list till depth i
#     x_d = {0: 0}
#     sum1 = 0.0
#     for i in range(l):
#         x = L[i]
#         y = S[i] if i < s else None
#         d = i + 1

#         # if two elements are same then
#         # we don't need to add to either of the set
#         if x == y:
#             x_d[d] = x_d[d - 1] + 1
#         # else add items to respective list
#         # and calculate overlap
#         else:
#             ls.add(x)
#             if y != None:
#                 ss.add(y)
#             x_d[d] = x_d[d - 1] + (1 if x in ss else 0) + (1 if y in ls else 0)
#         # calculate average overlap
#         sum1 += x_d[d] / d * pow(p, d)

#     sum2 = 0.0
#     for i in range(l - s):
#         d = s + i + 1
#         sum2 += x_d[d] * (d - s) / (d * s) * pow(p, d)

#     sum3 = ((x_d[l] - x_d[s]) / l + x_d[s] / s) * pow(p, l)

#     # Equation 32
#     rbo_ext = (1 - p) / p * (sum1 + sum2) + sum3
#     return rbo_ext


def rank_biased_overlap(ranks: pd.DataFrame, decay: float = 0.9) -> float:
    """
    Slightly limited home-made rank-biased overlap score.
    Input is a 2-col dataframe (ensuring consistent indices)
    This means there will always be a positive score due to the final term (all vs all)
    """

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


def rank_biased_overlap_weight(p: float, n: int) -> float:
    # infinite sum of p**i = 1 / (1 - p)
    return sum(p**i for i in range(n)) * (1 - p)
