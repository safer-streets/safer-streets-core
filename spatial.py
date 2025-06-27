from itertools import pairwise
from typing import Literal

import geopandas as gpd
import h3pandas  # noqa (implicitly required)
import numpy as np
import osmnx as ox
from shapely import Polygon, transform

from utils import Force

SpatialUnit = Literal["MSOA", "LSOA", "OA", "GRID", "HEX", "STREET"]
CensusGeography = Literal["MSOA", "LSOA", "OA"]
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
    boundaries = gpd.read_file(f"./data/{CENSUS_BOUNDARY_FILES[geography][resolution]}").set_index(f"{geography}CD")
    if overlapping is not None:
        # Drop boundaries that adjoin the overlapping area (but might overlap slightly due to rounding errors)
        joined = boundaries.sjoin(overlapping[["geometry"]], how="inner", predicate="intersects")
        # Calculate intersection area as a fraction of the boundary's area
        intersection = joined.geometry.intersection(overlapping.unary_union)
        # Drop any without significant overlap
        boundaries = joined[intersection.area / joined.geometry.area > 0.01].drop(columns=["index_right", "GlobalID"])
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

    grid = gpd.GeoDataFrame(geometry=p, crs="EPSG:27700").sjoin(boundary[["geometry"]]).drop(columns="index_right")
    return _add_centroids(grid)


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
    return _add_centroids(grid)


# not available in the police API...
def get_force_boundary(force_name: Force) -> gpd.GeoDataFrame:
    force_boundaries = gpd.read_file("./data/Police_Force_Areas_December_2023_EW_BFE_2734900428741300179.zip")
    if force_name not in force_boundaries.PFA23NM.to_list():
        raise ValueError(f"{force_name} is not valid. Must be one of {', '.join(force_boundaries.PFA23NM)}")
    return force_boundaries[force_name == force_boundaries.PFA23NM].drop(
        columns=["BNG_E", "BNG_N", "LAT", "LONG", "GlobalID"]
    )


def snap_to_street_segment(points: gpd.GeoDataFrame, street_segments: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Appends 2 columns to dataframe points"""
    map, dist = street_segments.geometry.sindex.nearest(points.geometry, return_distance=True, return_all=False)
    points.loc[:, "street_segment"] = street_segments.iloc[map[1]].index
    points.loc[:, "distance"] = dist
    return points


def map_to_spatial_unit(
    raw_crime_data: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame, area_type: SpatialUnit, **kwargs
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    crime_data = boundary[["geometry"]].sjoin(raw_crime_data, how="right").drop(columns="index_left")

    total_crimes = len(crime_data)

    # NOTE: in ~4% of (ASB in WY) cases the LSOA code is either inactive and/or does not contain the point
    # although the distance between the point and the reported LSOA is very small

    match area_type:
        case "LSOA":
            features = get_census_boundaries("LSOA21", overlapping=boundary, **kwargs)
            crime_data = features.sjoin(crime_data, how="right").rename(columns={"LSOA21CD": "spatial_unit"})
        case "MSOA":
            features = get_census_boundaries("MSOA21", overlapping=boundary, **kwargs)
            crime_data = features.sjoin(crime_data, how="right").rename(columns={"MSOA21CD": "spatial_unit"})
        case "OA":
            # do a spatial join to get ALL OAs in force area
            features = get_census_boundaries("OA21", overlapping=boundary, **kwargs)  # .sjoin(boundary[["geometry"]])
            # get crime counts for OAs by right-joining boundaries to crime data (this wont include crime-free OAs)
            crime_data = features.sjoin(crime_data, how="right").rename(columns={"OA21CD": "spatial_unit"})
        case "GRID":
            features = get_square_grid(boundary, **kwargs)
            crime_data = features.sjoin(crime_data, how="right").rename(columns={"index_left": "spatial_unit"})
        case "HEX":
            features = get_hex_grid(boundary, **kwargs)
            crime_data = features.sjoin(crime_data, how="right").rename(columns={"h3_polyfill": "spatial_unit"})
        case "STREET":
            # get street network in lon-lat polygon then project back to BNG
            G = ox.graph_from_polygon(boundary.to_crs(epsg=4326).iloc[0].geometry, network_type="drive")
            G = ox.project_graph(G, to_crs="epsg:27700")
            _nodes, features = ox.graph_to_gdfs(G)
            crime_data = snap_to_street_segment(crime_data, features).rename(columns={"street_segment": "spatial_unit"})

    # all crimes should be accounted for
    assert total_crimes == len(crime_data)

    return crime_data, features
