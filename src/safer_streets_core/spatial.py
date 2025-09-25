from itertools import pairwise
from typing import Any, Literal

import geopandas as gpd
import h3pandas  # noqa (implicitly required)
import numpy as np
import osmnx as ox
import pandas as pd
import shapely
from shapely import Polygon, transform

from safer_streets_core.utils import Force, data_dir, tokenize_force_name

SpatialUnit = Literal["MSOA21", "LSOA21", "OA21", "GRID", "H3", "HEX", "STREET"]
Resolution = Literal["FE", "GC", "SC"]

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


def _add_centroids(spatial_units: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    spatial_units["BNG_E"] = spatial_units.centroid.x
    spatial_units["BNG_N"] = spatial_units.centroid.y
    lonlat = spatial_units.centroid.to_crs(epsg=4326)
    spatial_units["LAT"] = lonlat.y
    spatial_units["LONG"] = lonlat.x
    return spatial_units


def get_census_boundaries(
    geography: str, *, resolution: Resolution = "GC", overlapping: gpd.GeoDataFrame | None = None
) -> gpd.GeoDataFrame:
    boundaries = gpd.read_file(data_dir() / f"{CENSUS_BOUNDARY_FILES[geography][resolution]}").set_index(
        f"{geography}CD"
    )
    boundaries.index.name = "spatial_unit"
    if overlapping is not None:
        # Drop boundaries that adjoin the overlapping area (but might overlap slightly due to rounding errors)
        joined = boundaries.sjoin(overlapping[["geometry"]], how="inner", predicate="intersects")
        # Calculate intersection area as a fraction of the boundary's area
        intersection = joined.geometry.intersection(overlapping.unary_union)
        # Drop any without significant overlap
        boundaries = joined[intersection.area / joined.geometry.area > 0.01].drop(
            columns=["index_right", "GlobalID"], errors="ignore"
        )
    return boundaries


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
        .drop(columns=[boundary.index.name, "index_right"], errors="ignore")
    )
    # trim features that cross the boundary (overlay removes the name of the index)
    grid = _add_centroids(grid).overlay(boundary)
    grid.index.name = "spatial_unit"
    return grid


def get_h3_grid(
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

    h3cells = (
        gpd.GeoDataFrame(geometry=boundary.geometry.buffer(2000))
        .to_crs(epsg=4326)
        .h3.polyfill_resample(resolution)
        .to_crs(epsg=27700)
    )

    if offset:
        h3cells.geometry = transform(h3cells.geometry, lambda xy: xy + offset)

    grid = (
        gpd.GeoDataFrame(geometry=h3cells.geometry, crs="EPSG:27700")
        .sjoin(boundary[["geometry"]])
        .drop(columns=[boundary.index.name, "index_right"], errors="ignore")
    )
    grid.index.name = "spatial_unit"
    # trim features that cross the boundary
    grid = _add_centroids(grid).overlay(boundary)
    grid.index.name = "spatial_unit"
    return grid


def get_hex_grid(
    boundary: gpd.GeoDataFrame, *, size: float, offset: tuple[float, float] | None = None
) -> gpd.GeoDataFrame:
    "size is the length of one side. The corresponds to an area of 3/2*sqrt(3)*s**2"

    xoff, yoff = offset or (0.0, 0.0)
    xmin, ymin, xmax, ymax = boundary.total_bounds

    dx = size * 3 / 2
    dy = size * np.sqrt(3)
    h = dy / 2

    def hexagon(x0, y0) -> list[tuple[float, float]]:
        "(x0, y0) is hexagon centre"
        return [
            (x0 + size, y0),
            (x0 + size / 2, y0 + h),
            (x0 - size / 2, y0 + h),
            (x0 - size, y0),
            (x0 - size / 2, y0 - h),
            (x0 + size / 2, y0 - h),
        ]

    X = np.arange(xmin // size * size - size + xoff, xmax // size * size + 3 * size + xoff, 2 * dx)
    Y = np.arange(ymin // size * size - size + yoff, ymax // size * size + 3 * size + yoff, dy)
    p = [Polygon(hexagon(x, y)) for x in X for y in Y]
    X = np.arange(xmin // size * size - size + xoff + dx, xmax // size * size + 3 * size + xoff + dx, 2 * dx)
    Y = np.arange(ymin // size * size - size + yoff + h, ymax // size * size + 3 * size + yoff + h, dy)
    p.extend([Polygon(hexagon(x, y)) for x in X for y in Y])

    grid = (
        gpd.GeoDataFrame(geometry=p, crs="EPSG:27700")
        .sjoin(boundary[["geometry"]])
        .drop(columns=[boundary.index.name, "index_right"], errors="ignore")
    )
    # trim features that cross the boundary
    grid = _add_centroids(grid).overlay(boundary)
    grid.index.name = "spatial_unit"
    return grid


def get_street_network(boundary: gpd.GeoDataFrame, *, network_type: str = "drive", **args: Any) -> gpd.GeoDataFrame:
    # TODO it would be good to cache this data but there are some issues with pyarrow/geojson
    # get street network in lon-lat polygon then project back to BNG
    G = ox.graph_from_polygon(
        boundary.to_crs(epsg=4326).iloc[0].geometry, network_type=network_type, retain_all=True, **args
    )
    G = ox.project_graph(G, to_crs="epsg:27700")
    _nodes, features = ox.graph_to_gdfs(G)

    def force_string(s: str | list[str]) -> str:
        return s if isinstance(s, str) else ",".join(s)

    features.highway = features.highway.map(force_string)
    return features


# not available in the police API...
def get_force_boundary(force_name: Force) -> gpd.GeoDataFrame:
    # correct for naming inconsistencies
    NAME_ADJUSTMENTS = {
        "Metropolitan": "Metropolitan Police",
        "Devon and Cornwall": "Devon & Cornwall",
        "City of London": "London, City of",
        "Dyfed Powys": "Dyfed-Powys",
    }
    corrected_force_name = NAME_ADJUSTMENTS.get(force_name, force_name)

    force_boundaries = gpd.read_file(data_dir() / "Police_Force_Areas_December_2023_EW_BFE_2734900428741300179.zip")
    if corrected_force_name not in force_boundaries.PFA23NM.to_list():
        raise ValueError(f"{corrected_force_name} is not valid. Must be one of {', '.join(force_boundaries.PFA23NM)}")
    return force_boundaries[corrected_force_name == force_boundaries.PFA23NM].drop(
        columns=["BNG_E", "BNG_N", "LAT", "LONG", "GlobalID"]
    )


def snap_to_street_segment(points: gpd.GeoDataFrame, street_segments: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Appends 2 columns to dataframe points"""
    map_, dist = street_segments.geometry.sindex.nearest(points.geometry, return_distance=True, return_all=False)
    points.loc[:, "street_segment"] = street_segments.iloc[map_[1]].index
    points.loc[:, "distance"] = dist
    return points


def map_to_spatial_unit(
    raw_crime_data: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame, area_type: SpatialUnit, **kwargs
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    crime_data = boundary[["geometry"]].sjoin(raw_crime_data, how="right").drop(columns="index_left")

    total_crimes = len(crime_data)

    # TODO crime_data may still contain points outside the boundary due to right sjoins?

    # NOTE: in ~4% of (ASB in WY) cases the LSOA code is either inactive and/or does not contain the point
    # although the distance between the point and the reported LSOA is very small

    match area_type:
        case "MSOA21" | "LSOA21" | "OA21":
            features = get_census_boundaries(area_type, overlapping=boundary, **kwargs)
            crime_data = features.sjoin(crime_data, how="right").rename(columns={f"{area_type}CD": "spatial_unit"})
        case "GRID":
            features = get_square_grid(boundary, **kwargs)
            crime_data = features.sjoin(crime_data, how="right").rename(columns={"index_left": "spatial_unit"})
        case "HEX":
            features = get_hex_grid(boundary, **kwargs)
            crime_data = features.sjoin(crime_data, how="right").rename(columns={"index_left": "spatial_unit"})
        case "H3":
            features = get_h3_grid(boundary, **kwargs)
            crime_data = features.sjoin(crime_data, how="right").rename(columns={"h3_polyfill": "spatial_unit"})
        case "STREET":
            features = get_street_network(boundary)
            crime_data = snap_to_street_segment(crime_data, features).rename(columns={"street_segment": "spatial_unit"})

    # all crimes should be accounted for
    assert total_crimes == len(crime_data)

    return crime_data, features


def normalised_clumpiness(features: gpd.GeoDataFrame, scale: float) -> float:
    u = features.union_all()

    if scale <= 0.0 or scale * scale > u.area:
        raise ValueError(f"Impossible scale parameter: {scale}. Bounds are (0, {np.sqrt(u.area)})")

    # compute the perimeter bounds
    max_p = 4 * u.area / scale  # all separate
    min_p = 4 * np.sqrt(u.area)  # single square

    # # TODO this needs testing
    # elif spatial_unit == "HEX": # TODO H3? irregular?
    #     max_p = 4 / np.sqrt(3) * u.area / scale
    #     min_p = np.sqrt(2 * u.area / (3 * np.sqrt(3)))

    r = max_p - min_p

    if max_p < u.length:
        # TODO and return 0?
        raise ValueError(f"Scale parameter {scale} is too large to capture features is the data.")

    # for a single unit clumpiness isnt defined but we return 1
    return (max_p - u.length) / r if r else 1.0


def load_population_data(force: Force) -> gpd.GeoDataFrame:
    """Loads a previously assigned point population dataset for a given force."""
    TABLE_NAME = "NM_2132_1"
    file = data_dir() / f"{TABLE_NAME}_assigned_{tokenize_force_name(force)}.parquet"
    if not file.exists():
        raise FileNotFoundError(f"Population data for {force} not found ({file}). Run assign-population first.")
    population = pd.read_parquet(file)
    population.geometry = shapely.from_wkt(population.geometry)
    return gpd.GeoDataFrame(population, crs="EPSG:27700")


def get_demographics(population: gpd.GeoDataFrame, features: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Returns a GeoDataFrame with population counts per spatial unit.
    Note units with zero population will be omitted. Reindexing is easier on a single index
    """
    remapped = features.sjoin(population)
    return (
        remapped.groupby(["spatial_unit", "C2021_ETH_20_NAME", "C2021_AGE_6_NAME", "C_SEX_NAME"], observed=False)
        .size()
        .rename("count")
        .to_frame()
        .sort_index()
    )
