from time import time

import geopandas as gpd
import pandas as pd

from safer_streets_core.spatial import get_hex_grid
from safer_streets_core.utils import (
    data_dir,
)


def main() -> None:
    """Script that generates 200m (side) hex cells over England & Wales"""
    t = time()
    # set up some parameters
    SPATIAL_UNIT = "HEX"
    SPATIAL_UNIT_SIZE = 200  # 200m side corresponds to ~350m height

    e_boundary = gpd.read_file(data_dir() / "E92000001.geojson")
    w_boundary = gpd.read_file(data_dir() / "W92000004.geojson")
    ew_boundary = gpd.GeoDataFrame(
        geometry=[pd.concat([e_boundary, w_boundary]).union_all()], crs=e_boundary.crs
    ).to_crs(epsg=27700)

    hexes = get_hex_grid(ew_boundary, size=SPATIAL_UNIT_SIZE, trim=False)
    print(f"t={time() - t:.1f}s")
    hexes.to_parquet(data_dir() / f"england_wales_{SPATIAL_UNIT}-{SPATIAL_UNIT_SIZE}_bb.parquet")
    print(len(hexes))


if __name__ == "__main__":
    main()
