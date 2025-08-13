import geopandas as gpd
import humanleague as hl
import numpy as np
import pandas as pd
import shapely
from tqdm import tqdm

from safer_streets_core import DATA_DIR
from safer_streets_core.nomisweb import TableMetadata, build_geog_query, fetch, fetch_table
from safer_streets_core.spatial import get_census_boundaries, get_force_boundary, get_street_network
from safer_streets_core.utils import Force, tokenize_force_name

table_name = "NM_2132_1"

def main(force: Force, *, seed: int=19937) -> None:

    boundary = get_force_boundary(force)

    features = get_census_boundaries("OA21", resolution="FE", overlapping=boundary)

    geog_lookup = pd.read_parquet(DATA_DIR / "census2021geographies.parquet").loc[features.index]
    nomis_area_codes = geog_lookup.NomisCode.to_list()

    print(f"Fetching metadata for {table_name}...")
    metadata = TableMetadata(**fetch(f"dataset/{table_name}.def.sdmx.json"))
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

    print(f"Fetching {table_name} data for {FORCE}...")
    data = fetch_table(table_name, **params)
    # using categories saves a ton of memory
    data.GEOGRAPHY_CODE = data.GEOGRAPHY_CODE.astype("category")
    data.C2021_ETH_20_NAME = data.C2021_ETH_20_NAME.astype("category")
    data.C2021_AGE_6_NAME = data.C2021_AGE_6_NAME.astype("category")
    data.C_SEX_NAME = data.C_SEX_NAME.astype("category")

    # expand to one row per person
    exploded = data.loc[np.repeat(data.index.values, data.OBS_VALUE.values)].drop(columns=["OBS_VALUE"]).reset_index(drop=True)
    # print(exploded.info())
    # print(exploded.memory_usage(deep=True))

    print(f"Fetching street network data for {FORCE}...")
    street_network = get_street_network(boundary)

    # TODO filter out non-residential streets
    # print(street_network.highway.value_counts())

    rng = np.random.default_rng(seed)  # for reproducibility
    exploded["geometry"] = None

    # TODO tqdm

    for oa_code, group in tqdm(list(exploded.groupby("GEOGRAPHY_CODE")), desc="Assigning population"):
        # print(f"Processing {oa_code} with population {len(group)}")

        try:
            local_streets = gpd.overlay(street_network, features.loc[[oa_code]])  #, how="intersection")

            # assign population proportionally to street segments
            pop_per_street, stats = hl.integerise(len(group) * local_streets.length / local_streets.length.sum())
            assert stats["conv"]
            # workaround a rounding error bug in humanleague
            if pop_per_street.sum() != len(group):
                # print(f"{oa_code} population mismatch: {pop_per_street.sum()} != {len(group)}")
                pop_per_street[pop_per_street.argmax()] += 1
            pop = local_streets.sample_points(pop_per_street, rng=rng).explode()  # need to explode to count intersections with individual points

            if len(group.index) != len(pop):
                raise ValueError(f"Population count mismatch for {oa_code}: {len(group.index)} != {len(pop)}")
            exploded.loc[group.index, "geometry"] = pop.geometry.to_numpy()
        except Exception as e:
            print(f"Error processing {oa_code}: {e}")
            continue

    # to avoid issues with geopandas and pyarrow/parquet, we use pandas to save the data
    exploded.geometry = shapely.to_wkt(exploded.geometry)
    exploded.to_parquet(DATA_DIR / f"{table_name}_assigned_{tokenize_force_name(force)}.parquet", index=False)

if __name__ == "__main__":
    FORCE = "West Yorkshire"
    # main(FORCE)

    population = pd.read_parquet(DATA_DIR / f"{table_name}_assigned_{tokenize_force_name(FORCE)}.parquet")
    population.geometry = shapely.from_wkt(population.geometry)
    population = gpd.GeoDataFrame(population, crs="EPSG:27700")

    print(population.geometry.isna().mean())
