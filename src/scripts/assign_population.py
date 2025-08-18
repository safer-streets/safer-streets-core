from typing import cast

import geopandas as gpd
import humanleague as hl
import numpy as np
import pandas as pd
import shapely
import typer
from tqdm import tqdm

from safer_streets_core import DATA_DIR
from safer_streets_core.nomisweb import TableMetadata, build_geog_query, fetch, fetch_table
from safer_streets_core.spatial import get_census_boundaries, get_force_boundary, get_street_network
from safer_streets_core.utils import Force, tokenize_force_name

# TODO hard-coded for now
TABLE_NAME = "NM_2132_1"


def impl(force: str, *, seed: int = 19937) -> None:
    force = cast(Force, force)  # type: ignore
    print(f"Fetching OA21 boundaries for {force}...")
    boundary = get_force_boundary(force)
    features = get_census_boundaries("OA21", resolution="FE", overlapping=boundary)

    geog_lookup = pd.read_parquet(DATA_DIR / "census2021geographies.parquet").loc[features.index]
    nomis_area_codes = geog_lookup.NomisCode.to_list()

    print(f"Fetching metadata for {TABLE_NAME}...")
    metadata = TableMetadata(**fetch(f"dataset/{TABLE_NAME}.def.sdmx.json"))
    fields = {
        a.conceptref: {"codelist": a.codelist} for a in metadata.structure.keyfamilies.keyfamily[0].components.dimension
    }

    selections = ",".join(
        (
            "GEOGRAPHY_CODE",
            *(f"{field}_NAME" for field in fields if field not in ["GEOGRAPHY", "FREQ", "MEASURES"]),
            "OBS_VALUE",
        )
    )
    params = {
        "date": "latest",
        "geography": build_geog_query(nomis_area_codes),
        "c2021_eth_20": "1001...1005",
        "c2021_age_6": "1...5",
        "c_sex": "1,2",
        "select": selections,
    }

    print(f"Fetching {TABLE_NAME} data for {force}...")
    data = fetch_table(TABLE_NAME, **params)
    # using categories saves a ton of memory
    data.GEOGRAPHY_CODE = data.GEOGRAPHY_CODE.astype("category")
    data.C2021_ETH_20_NAME = data.C2021_ETH_20_NAME.astype("category")
    data.C2021_AGE_6_NAME = data.C2021_AGE_6_NAME.astype("category")
    data.C_SEX_NAME = data.C_SEX_NAME.astype("category")

    # expand to one row per person
    exploded = (
        data.loc[np.repeat(data.index.values, data.OBS_VALUE.values)].drop(columns=["OBS_VALUE"]).reset_index(drop=True)
    )

    print(f"Fetching street network data for {force}...")
    street_network = get_street_network(boundary)

    # TODO filter out non-residential streets
    # print(street_network.highway.value_counts())

    rng = np.random.default_rng(seed)  # for reproducibility
    exploded["geometry"] = None

    for oa_code, group in tqdm(list(exploded.groupby("GEOGRAPHY_CODE", observed=True)), desc="Assigning population"):
        try:
            local_streets = gpd.overlay(street_network, features.loc[[oa_code]])  # , how="intersection")
        except ValueError as e:
            print(f"{oa_code} error {e}, falling back to sampling points in polygon")
            # TODO just sample points in polygon
            points = features.loc[[oa_code]].sample_points(len(group), rng=rng).explode()
            exploded.loc[group.index, "geometry"] = points.geometry.to_numpy()
            continue

        # assign population proportionally to street segments
        # a rounding error bug in humanleague means we need to explicitly specify the total
        # (and no conv status is returned for this overload)
        pop_per_street, stats = hl.integerise(
            len(group) * local_streets.length / local_streets.length.sum(), len(group)
        )
        # assert stats["conv"]
        points = local_streets.sample_points(
            pop_per_street, rng=rng
        ).explode()  # need to explode to count intersections with individual points

        if len(group.index) != len(points):
            raise ValueError(f"Population count mismatch for {oa_code}: {len(group.index)} != {len(points)}")
        exploded.loc[group.index, "geometry"] = points.geometry.to_numpy()

    # to avoid issues with geopandas and pyarrow/parquet, use pandas to save the data (need to convert geometry to WKT)
    exploded.geometry = shapely.to_wkt(exploded.geometry)
    exploded.to_parquet(DATA_DIR / f"{TABLE_NAME}_assigned_{tokenize_force_name(force)}.parquet", index=False)


def main() -> None:
    typer.run(impl)


if __name__ == "__main__":
    typer.run(main)
