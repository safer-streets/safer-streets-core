from typing import Literal

import geopandas as gpd
import osmnx as ox

from utils import get_census_boundaries, get_hex_grid, get_square_grid, snap_to_street_segment

SpatialUnit = Literal["MSOA", "LSOA", "OA", "GRID", "HEX", "STREET"]


def map_to_spatial_unit(
    raw_crime_data: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame, area_type: SpatialUnit
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    crime_data = boundary.sjoin(raw_crime_data, how="right").drop(columns="index_left")

    total_crimes = len(crime_data)

    # NOTE: in ~4% of (ASB in WY) cases the LSOA code is either inactive and/or does not contain the point
    # although the distance between the point and the reported LSOA is very small

    match area_type:
        case "LSOA":
            features = get_census_boundaries("LSOA21", "GC", overlapping=boundary)
            crime_data = features.sjoin(crime_data, how="right").rename(columns={"LSOA21CD": "spatial_unit"})
        case "MSOA":
            features = get_census_boundaries("MSOA21", "GC", overlapping=boundary)
            crime_data = features.sjoin(crime_data, how="right").rename(columns={"MSOA21CD": "spatial_unit"})
        case "OA":
            # do a spatial join to get ALL OAs in force area
            features = get_census_boundaries("OA21", "GC", overlapping=boundary)  # .sjoin(boundary[["geometry"]])
            # get crime counts for OAs by right-joining boundaries to crime data (this wont include crime-free OAs)
            crime_data = features.sjoin(crime_data, how="right").rename(columns={"OA21CD": "spatial_unit"})
        case "GRID":
            features = get_square_grid(500.0, boundary)
            crime_data = features.sjoin(crime_data, how="right").rename(columns={"index_left": "spatial_unit"})
        case "HEX":
            features = get_hex_grid(8, boundary)
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
