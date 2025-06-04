from collections.abc import Iterator
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd
from shapely import Polygon

CATEGORIES = ("Violence and sexual offences", "Public order", "Possession of weapons")


def format_boundary_as_param(polygon: Polygon) -> str:
    xy = zip(*(c.tolist() for c in polygon.exterior.coords.xy))
    return ":".join(f"{x:.3f},{y:.3f}" for x, y in xy)


# Ensure you download at least one of these from ONS
# e.g. https://geoportal.statistics.gov.uk/datasets/ons::lower-layer-super-output-areas-december-2021-boundaries-ew-bsc-v4-2/about
LSOA_BOUNDARY_FILES = {
    "FE": "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BFE_V10_-3435351624505741073.zip",
    "GC": "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BGC_V5_4492169359079898015.zip",
    "SC": "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BSC_V4_-5236167991066794441.zip",
}


def get_lsoa_boundaries(resolution: str, *, overlapping: gpd.GeoDataFrame | None = None) -> gpd.GeoDataFrame:
    lsoa_boundaries = gpd.read_file(f"./data/{LSOA_BOUNDARY_FILES[resolution]}").set_index("LSOA21CD")
    if overlapping is not None:
        # throw away any not in the bounding box defined by the crimes
        bbox = overlapping.geometry.union_all()  # .envelope
        lsoa_boundaries = lsoa_boundaries[lsoa_boundaries.geometry.intersects(bbox)]
    return lsoa_boundaries


# using https://data.police.uk/data/, "custom download" tab
def extract_crime_data(path: Path | str) -> gpd.GeoDataFrame:
    with ZipFile(path) as bulk_data:
        crimes = []
        outcomes = []
        for file in bulk_data.namelist():
            if "street" in file:
                crimes.append(pd.read_csv(bulk_data.open(file)))
            elif "outcomes" in file:
                outcomes.append(pd.read_csv(bulk_data.open(file)))
        crime_data = pd.concat(crimes).set_index("Crime ID")
        outcome_data = pd.concat(outcomes).set_index("Crime ID")

    # outcomes only differ for <10% of crimes
    crime_data = crime_data.merge(outcome_data["Outcome type"], left_index=True, right_index=True)

    # drop crimes with no location
    crime_data = crime_data.dropna(subset=["Longitude", "Latitude"])

    return gpd.GeoDataFrame(
        crime_data.drop(columns=["Longitude", "Latitude"]),
        geometry=gpd.points_from_xy(crime_data.Longitude, crime_data.Latitude),
        crs="EPSG:4326",
    ).to_crs(epsg=27700)


def monthgen(year: int, month: int) -> Iterator[str]:
    while True:
        yield f"{year}-{month:02}"
        month += 1
        if month == 13:
            month = 1
            year += 1
