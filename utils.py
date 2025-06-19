from __future__ import annotations

from collections.abc import Iterator
from functools import cache
from itertools import pairwise
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import h3pandas  # noqa (implicitly required)
import numpy as np
import numpy.typing as npt
import pandas as pd
import requests
from shapely import Polygon, transform

from models import Neighbourhoods, RawPolygon

CATEGORIES = ("Violence and sexual offences", "Anti-social behaviour", "Possession of weapons")


def format_boundary_as_param(polygon: Polygon) -> str:
    xy = zip(*(c.tolist() for c in polygon.exterior.coords.xy))
    return ":".join(f"{x:.3f},{y:.3f}" for x, y in xy)


# Download at least one of these from ONS
# e.g. https://geoportal.statistics.gov.uk/datasets/ons::lower-layer-super-output-areas-december-2021-boundaries-ew-bsc-v4-2/about
CENSUS_BOUNDARY_FILES = {
    "MSOA21": {
        "FE": "Middle_layer_Super_Output_Areas_December_2021_Boundaries_EW_BFE_V8_-1517080999235121072.zip",
        "GC": "Middle_layer_Super_Output_Areas_December_2021_Boundaries_EW_BGC_V3_-6221323399304446140.zip",
    },
    "LSOA21": {
        "FE": "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BFE_V10_-3435351624505741073.zip",
        "GC": "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BGC_V5_4492169359079898015.zip",
        "SC": "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BSC_V4_-5236167991066794441.zip",
    },
    "OA21": {
        "FE": "Output_Areas_2021_EW_BFE_V9_-4280877107876255952.zip",
        "GC": "Output_Areas_2021_EW_BGC_V2_-6371128854279904124.zip",
    },
}


def get_census_boundaries(
    geography: str, resolution: str, *, overlapping: gpd.GeoDataFrame | None = None
) -> gpd.GeoDataFrame:
    boundaries = gpd.read_file(f"./data/{CENSUS_BOUNDARY_FILES[geography][resolution]}").set_index(f"{geography}CD")
    if overlapping is not None:
        # Drop boundaries that adjoin the overlapping area (but might overlap slightly due to rounding errors)
        joined = boundaries.sjoin(overlapping, how="inner", predicate="intersects")
        # Calculate intersection area as a fraction of the boundary's area
        intersection = joined.geometry.intersection(overlapping.unary_union)
        # Drop any without significant overlap
        boundaries = joined[intersection.area / joined.geometry.area > 0.01].drop(columns=["index_right"])
    return boundaries


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


# not available in the police API...
def get_force_boundary(force_name: str) -> gpd.GeoDataFrame:
    force_boundaries = gpd.read_file("./data/Police_Force_Areas_December_2023_EW_BFE_2734900428741300179.zip")
    if force_name not in force_boundaries.PFA23NM.to_list():
        raise ValueError(f"{force_name} is not valid. Must be one of {', '.join(force_boundaries.PFA23NM)}")
    return force_boundaries[force_name == force_boundaries.PFA23NM].drop(
        columns=["BNG_E", "BNG_N", "LAT", "LONG", "GlobalID"]
    )


def get_square_grid(
    boundary: gpd.GeoDataFrame, *, size: float, offset: tuple[float, float] = (0.0, 0.0)
) -> gpd.GeoDataFrame:
    assert (-size, -size) < offset < (size, size), "offsets should be smaller than size"

    xoff, yoff = offset
    xmin, ymin, xmax, ymax = boundary.total_bounds

    # X = np.arange(xmin // size * size, xmax // size * (size + 1), size)
    X = np.arange(xmin // size * size - size + xoff, xmax // size * size + 3 * size + xoff, size)
    Y = np.arange(ymin // size * size - size + yoff, ymax // size * size + 3 * size + yoff, size)

    p = [Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]) for x0, x1 in pairwise(X) for y0, y1 in pairwise(Y)]

    grid = (
        gpd.GeoDataFrame(geometry=p, crs="EPSG:27700")
        .sjoin(boundary[["geometry"]]) 
        .drop(columns="index_right")
    )
    return grid


def get_hex_grid(
    boundary: gpd.GeoDataFrame, *, resolution: int, offset: tuple[float, float] | None = None
) -> gpd.GeoDataFrame:
    """
    Use resolution = 7 for ~4.5km2 cells, 8 for ~0.65km2 cells, 9 for ~0.1km2 cells
    h3_polyfill uses centroids to determine overlap so we add a 2km buffer then spatially join to original boundary
    Offset is metres (BNG)
    """
    # to offset hex grid without causing overlap mismatches:
    # shift boundary by -offset -> get hex grid -> shift grid by offset
    if offset:
        boundary.geometry = transform(boundary.geometry, lambda xy: xy - offset)

    hex = (
        gpd.GeoDataFrame(geometry=boundary.geometry.buffer(2000))
        .to_crs(epsg=4326)
        .h3.polyfill_resample(resolution)
        .to_crs(epsg=27700)
    )

    if offset:
        hex.geometry = transform(hex.geometry, lambda xy: xy + offset)

    grid = (
        gpd.GeoDataFrame(geometry=hex.geometry, crs="EPSG:27700")
        .sjoin(boundary[["geometry"]]) 
        .drop(columns="index_right")
    )
    return grid


def snap_to_street_segment(points: gpd.GeoDataFrame, street_segments: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Appends 2 columns to dataframe points"""
    map, dist = street_segments.geometry.sindex.nearest(points.geometry, return_distance=True, return_all=False)
    points.loc[:, "street_segment"] = street_segments.iloc[map[1]].index
    points.loc[:, "distance"] = dist
    return points


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
        crime_data = pd.concat(crimes).set_index("Crime ID")

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


def calc_gini(data: pd.Series[int]) -> tuple[float, npt.NDArray]:
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

    return _spearman_rank_correlation_impl(left_ranks - right_ranks)


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
