"""
Batch version of area-ranking.ipynb notebook
"""

from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from spatial import SpatialUnit, get_force_boundary, map_to_spatial_unit
from utils import (
    CATEGORIES,
    calc_gini,
    extract_crime_data,
    lorenz_curve,
    rank_biased_overlap,
    spearman_rank_correlation,
    tokenize_force_name,
)

FORCE = "West Yorkshire"
CATEGORY = CATEGORIES[1]  # ASB
AREA_PARAMETERS = defaultdict(
    dict,
    {
        "LSOA": {"resolution": "FE"},
        "GRID": {"size": 500.0},
        "HEX": {"resolution": 8},
    },
)
SAMPLE_FRACTION = 0.5


def calc_stats(counts: pd.Series):
    total_areas = len(counts)
    full_lorenz = lorenz_curve(counts)
    gini, lorenz = calc_gini(counts)  # .sort_values().cumsum() / counts.sum()
    return pd.Series(
        data={
            "prop_zero": (counts == 0).mean(),
            "prop_threshold": (full_lorenz >= SAMPLE_FRACTION).sum() / total_areas,
            "gini": gini,
            "lorenz": lorenz.to_numpy(),
        },
        name=counts.name,
    )


def plot(
    data_period: str,
    spatial_unit: SpatialUnit,
    window: int,
    stats: pd.DataFrame,
    month_on_month: pd.DataFrame,
    spatial_units: gpd.GeoDataFrame,
    total_counts: pd.DataFrame,
) -> Figure:
    fig, axs = plt.subplots(2, 2, figsize=(18, 15))
    fig.suptitle(f"{FORCE} - {CATEGORY} - {data_period} - {window} month rolling mean count per {spatial_unit}")
    # plt.tight_layout()
    stats.prop_zero.plot(ax=axs[0, 0], label="Zero count areas")
    stats.prop_threshold.plot(ax=axs[0, 0], label=f"Areas comprising {SAMPLE_FRACTION:.0%} of total crime")
    stats.gini.plot(ax=axs[0, 0], label="Gini coefficient")
    axs[0, 0].tick_params(axis="x", rotation=45)
    axs[0, 0].set_ylabel("Proportion of areas")
    axs[0, 0].legend()

    for name, row in stats.iterrows():
        axs[0, 1].plot(range(101), row["lorenz"], label=f"{name}")
    axs[0, 1].set_xlabel("Cumulative proportion of areas (%)")
    axs[0, 1].set_ylabel("Cumulative proportion of crime")
    axs[0, 1].set_title("Lorenz curves")

    month_on_month.plot(ax=axs[1, 0])
    axs[1, 0].set_title("Month-on-month comparison")
    axs[1, 0].tick_params(axis="x", rotation=45)

    if spatial_unit != "STREET":
        spatial_units.join(total_counts).fillna(0).plot(ax=axs[1,1], column="total_counts", alpha=1, legend=True)
        axs[1,1].set_title("Total crimes per area")
    axs[1,1].set_axis_off()

    return fig


def main() -> None:
    force_boundary = get_force_boundary(FORCE)
    raw_crime_data = extract_crime_data(tokenize_force_name(FORCE))
    # filter by catgegory and remove any points
    raw_crime_data = raw_crime_data[raw_crime_data["Crime type"] == CATEGORY]
    raw_crime_data = raw_crime_data[raw_crime_data.geometry.intersects(force_boundary.geometry.union_all())]
    print("Crime_data loaded")

    months = sorted(raw_crime_data.Month.unique())
    data_period = f"{months[0]} to {months[-1]}"

    smoothing_windows = [1, 3, 6, 12]
    spatial_unit_types = ["MSOA", "LSOA", "OA", "GRID", "HEX", "STREET"]

    for spatial_unit_type in spatial_unit_types:
        for window in smoothing_windows:
            filename = Path(f"{tokenize_force_name(FORCE)}-ASB-{spatial_unit_type}-{window}.png")
            if filename.exists():
                continue
            print(spatial_unit_type, window)
            crime_data, spatial_units = map_to_spatial_unit(
                raw_crime_data, force_boundary, spatial_unit_type, **AREA_PARAMETERS[spatial_unit_type]
            )

            # ensure we account for crime-free spatial units in the data
            counts = (
                crime_data.groupby(["Month", "spatial_unit"])["Crime type"]
                .count()
                .unstack(level="Month", fill_value=0)
                .sort_index()
            )
            total_counts = crime_data.groupby("spatial_unit").size().rename("total_counts")

            smoothed_counts = counts.T.rolling(window).mean().dropna().T
            ranks = smoothed_counts.apply(lambda col: col.rank(method="min", ascending=False))

            month_on_month = pd.DataFrame(index=pd.MultiIndex.from_tuples(pairwise(ranks.columns)))
            orderings = {colname: col.sort_values(ascending=False).index for colname, col in ranks.items()}

            month_on_month["Spearman rank correlation"] = month_on_month.index.map(
                lambda idx, ranks=ranks: spearman_rank_correlation(ranks[idx[0]], ranks[idx[1]])
            )
            month_on_month["Rank-biased overlap"] = month_on_month.index.map(
                lambda idx, orderings=orderings: rank_biased_overlap(orderings[idx[0]], orderings[idx[1]])
            )

            stats = pd.concat([calc_stats(series) for _, series in smoothed_counts.items()], axis=1).T

            fig = plot(data_period, spatial_unit_type, window, stats, month_on_month, spatial_units, total_counts)

            fig.savefig(filename)
            # plt.show()


if __name__ == "__main__":
    main()
