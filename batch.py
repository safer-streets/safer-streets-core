"""
Batch version of area-ranking.ipynb notebook
"""

from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from spatial import SpatialUnit, get_force_boundary, map_to_spatial_unit
from utils import (
    CATEGORIES,
    calc_gini,
    cosine_similarity,
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
    mean_rates: pd.DataFrame,
) -> Figure:
    fig, axs = plt.subplots(2, 2, figsize=(18, 15))
    fig.suptitle(f"{FORCE} - {CATEGORY} - {data_period} - {window} month rolling mean count per {spatial_unit}")
    # plt.tight_layout()
    stats.prop_zero.plot(ax=axs[0, 0], label="Zero count areas")
    stats.prop_threshold.plot(ax=axs[0, 0], label=f"Areas comprising {SAMPLE_FRACTION:.0%} of total crime")
    stats.gini.plot(ax=axs[0, 0], label="Gini coefficient")
    axs[0, 0].tick_params(axis="x", rotation=45)
    axs[0, 0].set_ylabel("Proportion of areas")
    axs[0, 0].set_ylim((-0.05, 1.05))
    axs[0, 0].legend()

    for name, row in stats.iterrows():
        axs[0, 1].plot(range(101), row["lorenz"], label=f"{name}")
    axs[0, 1].set_xlabel("Cumulative proportion of areas (%)")
    axs[0, 1].set_ylabel("Cumulative proportion of crime")
    axs[0, 1].set_title("Lorenz curves")

    month_on_month.plot(ax=axs[1, 0])
    axs[1, 0].set_title("Month-on-month comparison")
    axs[1, 0].tick_params(axis="x", rotation=45)
    axs[0, 0].set_ylim((-0.05, 1.05))

    if spatial_unit == "STREET":
        mean_rates.index = pd.MultiIndex.from_tuples(mean_rates.index, names=["u", "v", "key"])
    spatial_units.join(mean_rates).fillna(0).plot(ax=axs[1, 1], column="rate", alpha=1, legend=True)
    axs[1, 1].set_title("Mean crime rate per km² per year")
    axs[1, 1].set_axis_off()

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

    timespan = len(months) / 12

    smoothing_windows = [1, 3, 6, 12]
    spatial_unit_types: list[SpatialUnit] = ["MSOA", "LSOA", "OA", "GRID", "HEX", "STREET"]

    for spatial_unit_type in spatial_unit_types:
        crime_data, spatial_units = map_to_spatial_unit(
            raw_crime_data, force_boundary, spatial_unit_type, **AREA_PARAMETERS[spatial_unit_type]
        )

        for window in smoothing_windows:
            filename = Path(f"./output/{tokenize_force_name(FORCE)}-ASB-{spatial_unit_type}-{window}.png")
            # if filename.exists():
            #     continue
            print(spatial_unit_type, window)

            # ensure we account for crime-free spatial units in the data
            counts = (
                crime_data.groupby(["Month", "spatial_unit"])["Crime type"]
                .count()
                .unstack(level="Month", fill_value=0)
                .sort_index()
            )

            # compute rates
            if spatial_unit_type == "STREET":
                # proxy area as length * a width factor thats consistent with the force area
                width_factor = force_boundary.area.sum() / spatial_units.length.sum()
                areas = counts.index.map(spatial_units.length) * width_factor
            else:
                areas = counts.index.map(spatial_units.area)
            rates = counts.div(areas, axis=0) / timespan * 1_000_000  # crimes/sq.km/year

            smoothed_counts = counts.T.rolling(window).mean().dropna().T
            count_ranks = smoothed_counts.apply(lambda col: col.rank(ascending=False))
            smoothed_rates = rates.T.rolling(window).mean().dropna().T
            rate_ranks = smoothed_rates.apply(lambda col: col.rank(ascending=False))

            mean_rates = rates.mean(axis=1).rename("rate")

            month_on_month = pd.DataFrame(index=pd.MultiIndex.from_tuples(pairwise(count_ranks.columns)))

            month_on_month["Spearman rank correlation (count)"] = month_on_month.index.map(
                lambda idx, count_ranks=count_ranks: spearman_rank_correlation(count_ranks[list(idx)])
            )
            month_on_month["Spearman rank correlation (rate)"] = month_on_month.index.map(
                lambda idx, rate_ranks=rate_ranks: spearman_rank_correlation(rate_ranks[list(idx)])
            )
            month_on_month["Rank-biased overlap (count)"] = month_on_month.index.map(
                lambda idx, count_ranks=count_ranks: rank_biased_overlap(count_ranks[list(idx)])
            )
            month_on_month["Rank-biased overlap (rate)"] = month_on_month.index.map(
                lambda idx, rate_ranks=rate_ranks: rank_biased_overlap(rate_ranks[list(idx)])
            )
            month_on_month["Cosine similarity (count)"] = month_on_month.index.map(
                lambda idx, smoothed_counts=smoothed_counts: cosine_similarity(smoothed_counts[list(idx)])
            )
            month_on_month["Cosine similarity (rate)"] = month_on_month.index.map(
                lambda idx, smoothed_rates=smoothed_rates: cosine_similarity(smoothed_rates[list(idx)])
            )

            stats = pd.concat([calc_stats(series) for _, series in smoothed_counts.items()], axis=1).T

            fig = plot(data_period, spatial_unit_type, window, stats, month_on_month, spatial_units, mean_rates)

            fig.savefig(filename)
            # plt.show()


if __name__ == "__main__":
    main()
