from pathlib import Path
from zipfile import ZipFile

from shapely import Polygon

import geopandas as gpd
import pandas as pd


def format_boundary_as_param(polygon: Polygon) -> str:
    xy = zip(*(c.tolist() for c in polygon.exterior.coords.xy))
    return ":".join(f"{x:.3f},{y:.3f}" for x, y in xy)


# using https://data.police.uk/data/, "custom download" tab
def extract_crime_data(path: Path) -> gpd.GeoDataFrame:
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
    crime_data = crime_data.merge(
        outcome_data["Outcome type"], left_index=True, right_index=True
    )

    # drop crimes with no location
    crime_data = crime_data.dropna(subset=["Longitude", "Latitude"])

    return gpd.GeoDataFrame(
        crime_data.drop(columns=["Longitude", "Latitude"]),
        geometry=gpd.points_from_xy(crime_data.Longitude, crime_data.Latitude),
        crs="EPSG:4326",
    ).to_crs(epsg=27700)
