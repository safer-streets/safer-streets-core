from __future__ import annotations

from collections.abc import Iterator
from functools import cache, reduce
from itertools import zip_longest
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import requests
from shapely import Polygon

from models import Neighbourhoods, RawPolygon

CATEGORIES = ("Violence and sexual offences", "Anti-social behaviour", "Possession of weapons")


def format_boundary_as_param(polygon: Polygon) -> str:
    xy = zip(*(c.tolist() for c in polygon.exterior.coords.xy))
    return ":".join(f"{x:.3f},{y:.3f}" for x, y in xy)


def _get_boundary(force: str, neighbourhood_id: str) -> Polygon:
    try:
        return RawPolygon(
            requests.get(f"{POLICE_API_BASE_URL}/{force}/{neighbourhood_id}/boundary").json()
        ).to_shapely()
    except Exception:
        return Polygon()


@cache
def get_raw_geog_lookup() -> pd.DataFrame:
    # https://www.arcgis.com/sharing/rest/content/items/80592949bebd4390b2cbe29159a75ef4/data
    return pd.read_csv("./data/PCD_OA21_LSOA21_MSOA21_LAD_FEB25_UK_LU.zip")


def get_geog_lookup(geog_from: str, geogs_to: list[str]) -> pd.DataFrame:
    lookup = get_raw_geog_lookup()[[geog_from, *geogs_to]].drop_duplicates()
    return lookup.set_index(geog_from)


POLICE_API_BASE_URL = "http://data.police.uk/api"
POLICE_DATA_BASE_URL = "http://data.police.uk/data"
CRIME_ARCHIVE = Path("./data/police_data_latest.zip")


def tokenize_force_name(force_name: str) -> str:
    """Tokenize the force name for use in file paths."""
    return force_name.replace(" ", "-").lower()


def get_neighbourhood_boundaries(force: str) -> gpd.GeoDataFrame:
    neighbourhoods = Neighbourhoods(requests.get(f"{POLICE_API_BASE_URL}/{force}/neighbourhoods").json())
    neighbourhood_boundaries = gpd.GeoDataFrame(
        index=tuple(n.id for n in neighbourhoods),
        data={"name": (n.name for n in neighbourhoods)},
        geometry=[_get_boundary(force, n.id) for n in neighbourhoods],
        crs="epsg:4326",
    ).to_crs(epsg=27700)
    return neighbourhood_boundaries


def extract_crime_data(force: str, *, keep_lonlat: bool = False) -> gpd.GeoDataFrame:
    """
    Extracts crime data for a given force from the latest archive.
    Use keep_lonlat for rendering  streamlit maps
    """
    if not CRIME_ARCHIVE.exists():
        get_latest_archive()

    with ZipFile(CRIME_ARCHIVE) as bulk_data:
        crimes = []
        for file in bulk_data.namelist():
            if f"{force}-street" in file:
                crimes.append(pd.read_csv(bulk_data.open(file)))
        crime_data = pd.concat(crimes).set_index("Crime ID").drop(columns=["Last outcome category", "Context"])

    # drop crimes with no location
    crime_data = crime_data.dropna(subset=["Longitude", "Latitude"])

    return gpd.GeoDataFrame(
        crime_data.rename(columns={"Longitude": "lon", "Latitude": "lat"})
        if keep_lonlat
        else crime_data.drop(columns=["Longitude", "Latitude"]),
        geometry=gpd.points_from_xy(crime_data.Longitude, crime_data.Latitude),
        crs="EPSG:4326",
    ).to_crs(epsg=27700)


def get_latest_archive() -> bool:
    """
    Downloads the latest police data archive and saves it to CRIME_ARCHIVE.
    To force a download, delete the existing CRIME_ARCHIVE file, or just explicitly call this function.
    """
    url = f"{POLICE_DATA_BASE_URL}/archive/latest.zip"
    MB = 1024 * 1024

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        size = int(response.headers.get("content-length", 0))
        print(f"Downloading archive {size // MB}MB", end="")

        with open(CRIME_ARCHIVE, "wb") as f:
            for chunk in response.iter_content(chunk_size=MB):
                if chunk:
                    f.write(chunk)
                    print(".", end="", flush=True)
        print("\nDownload complete.")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")
    except OSError as e:
        print(f"Error writing file {CRIME_ARCHIVE}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return False


def monthgen(year: int, month: int) -> Iterator[str]:
    while True:
        yield f"{year}-{month:02}"
        month += 1
        if month == 13:
            month = 1
            year += 1


def lorenz_curve(data: pd.Series[int], *, percentiles: bool = False) -> pd.Series[float]:
    full = data.sort_values().cumsum() / data.sum()
    if percentiles:
        x = np.linspace(0, 100, 101)
        return pd.Series(index=x, data=np.percentile(full, x))
    return full


def calc_gini(data: pd.Series[int]) -> tuple[float, pd.Series]:
    lorenz = lorenz_curve(data, percentiles=True)
    # trapezoidal rule (scaled by 100 - x axis is %)
    gini = 1.0 - lorenz.rolling(2).mean().sum() / 50.0
    return gini, lorenz


def _spearman_rank_correlation_impl(diff: pd.Series) -> float:
    n = len(diff)
    return 1 - 6 * (diff**2).sum() / (n * (n * n - 1))


# this is inefficient for multiple correlations - see below
def spearman_rank_correlation(left: pd.Series, right: pd.Series) -> float:
    assert not len(np.setdiff1d(left.index, right.index)) and not len(np.setdiff1d(right.index, left.index))

    left_ranks = left.rank(method="min", ascending=False)
    right_ranks = right.rank(method="min", ascending=False)

    return _spearman_rank_correlation_impl(left - right)


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

    ranks = counts.apply(lambda col: col.rank(method="min", ascending=False))

    for i in range(n):
        for j in range(i):
            correlations[i, j] = correlations[j, i] = _spearman_rank_correlation_impl(
                ranks.iloc[:, i] - ranks.iloc[:, j]
            )
    return correlations


# based on code from https://towardsdatascience.com/rbo-v-s-kendall-tau-to-compare-ranked-lists-of-items-8776c5182899/
def rank_biased_overlap1(left: list[Any], right: list[Any], p: float = 0.9) -> float:
    k = max(len(left), len(right))
    x_k = len(set(left).intersection(right))
    summation = sum(p**i * len(set(left[:i]).intersection(right[:i])) / i for i in range(1, k + 1))
    return (float(x_k) / k * p**k) + ((1 - p) / p * summation)


def rbo_weight(p: float, n: int) -> float:
    """
    Contribution of top n rankings for a given decay constant p
    e.g. rbo_weight(0.9, 10) ~= 0.856
    """
    assert 0 < p <= 1
    s = sum(p**i / i for i in range(1, n))
    return 1.0 - p ** (n - 1) + ((1 - p) / p * n * (np.log(1 / (1 - p)) - s))


def rank_biased_overlap2(l1, l2, p=0.9):
    """
    Calculates Ranked Biased Overlap (RBO) score.
    l1 -- Ranked List 1
    l2 -- Ranked List 2
    """
    l1 = l1 or []
    l2 = l2 or []

    sl, ll = sorted([(len(l1), l1), (len(l2), l2)])
    s, S = sl
    l, L = ll
    if s == 0:
        return 0

    # Calculate the overlaps at ranks 1 through l
    # (the longer of the two lists)
    ss = set([])  # contains elements from the smaller list till depth i
    ls = set([])  # contains elements from the longer list till depth i
    x_d = {0: 0}
    sum1 = 0.0
    for i in range(l):
        x = L[i]
        y = S[i] if i < s else None
        d = i + 1

        # if two elements are same then
        # we don't need to add to either of the set
        if x == y:
            x_d[d] = x_d[d - 1] + 1
        # else add items to respective list
        # and calculate overlap
        else:
            ls.add(x)
            if y != None:
                ss.add(y)
            x_d[d] = x_d[d - 1] + (1 if x in ss else 0) + (1 if y in ls else 0)
        # calculate average overlap
        sum1 += x_d[d] / d * pow(p, d)

    sum2 = 0.0
    for i in range(l - s):
        d = s + i + 1
        sum2 += x_d[d] * (d - s) / (d * s) * pow(p, d)

    sum3 = ((x_d[l] - x_d[s]) / l + x_d[s] / s) * pow(p, l)

    # Equation 32
    rbo_ext = (1 - p) / p * (sum1 + sum2) + sum3
    return rbo_ext


# def rank_biased_overlap(col1: pd.Series, col2: pd.Series, decay: float = 0.9) -> float:
#     """
#     Slightly limited home-made rank-biased overlap score.
#     col1 and col2 must have the same indices (ordering can be different)
#     """

#     assert col1.index.equals(col2.index)
#     r1 = tuple(set(group.index) for _, group in col1.groupby(col1))
#     r2 = tuple(set(group.index) for _, group in col2.groupby(col2))

#     n = max(len(r1), len(r2))
#     if n == 1:
#         return 1.0

#     num = 0.0
#     # can skip the final term since the sets will be identical - add this at the end
#     for i in range(n - 1):
#         left = reduce(set.__or__, r1[: i + 1])
#         right = reduce(set.__or__, r2[: i + 1])
#         num += decay**i * 2 * len(left & right) / (len(left) + len(right))
#         # print(i, len(left & right), (len(left) + len(right))/2, num)
#     return (num + decay ** (n - 1)) / sum(decay**i for i in range(n))



def rank_biased_overlap(col1: pd.Series, col2: pd.Series, decay: float = 0.9) -> float:
    """
    Slightly limited home-made rank-biased overlap score.
    col1 and col2 must have the same indices (ordering can be different)
    This means there will always be a positive score due to the final term
    """

    assert col1.index.equals(col2.index)
    left_sets = tuple(set(group.index) for _, group in col1.groupby(col1))
    right_sets = tuple(set(group.index) for _, group in col2.groupby(col2))

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
        num += decay ** i * len(intersection) / len(union)
        den += decay ** i
    return num / den

def rank_biased_overlap_weight(p: float, n: int) -> float:
    # infinite sum of p**i = 1 / (1 - p)
    return sum(p ** i for i in range(n)) * (1 - p)

def cosine_similarity(col1, col2):
    return (col1 @ col2) / np.sqrt((col1 @ col1) * (col2 @ col2))
