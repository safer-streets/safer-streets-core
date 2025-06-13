from itertools import islice

import geopandas as gpd
import pandas as pd
import streamlit as st

from utils import (
    CATEGORIES,
    extract_crime_data,
    get_census_boundaries,
    get_force_boundary,
    monthgen,
    tokenize_force_name,
)


@st.cache_data
def load_crime_data(force: str, category: str) -> pd.DataFrame:
    force_boundary = get_force_boundary(force)
    data = extract_crime_data(tokenize_force_name(force), keep_lonlat=True)

    data = data[(data["Crime type"] == category)]  # & (data.geometry.intersects(force_boundary.geometry.union_all()))]
    return data, force_boundary


@st.cache_data
def load_lsoa_boundaries(_force_boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    lsoa_boundaries = get_census_boundaries("LSOA21", "GC", overlapping=_force_boundary)
    return lsoa_boundaries


st.set_page_config(layout="wide", page_title="Safer Streets", page_icon="👮")


def main() -> None:
    # TODO pickers
    force = "West Yorkshire"

    st.title(f"Crime patterns in {force}")

    category = st.selectbox("Crime type", CATEGORIES)

    all_months = list(islice(monthgen(2022, 5), 36))

    data, force_boundary = load_crime_data(force, category)
    lsoa_boundaries = load_lsoa_boundaries(force_boundary)

    st.toast("Data loaded")

    start_month, end_month = st.select_slider(
        "Select time range", options=all_months, value=(all_months[0], all_months[-1])
    )

    months = all_months[all_months.index(start_month) : all_months.index(end_month) + 1]
    data = data[data.Month.isin(months)]

    counts = (
        data.groupby(["Month", "LSOA code"])["Crime type"].count().unstack(level="Month", fill_value=0).sort_index()
    )
    counts = counts.reindex(lsoa_boundaries.index, fill_value=0)
    _num_areas = len(lsoa_boundaries)

    if st.checkbox("Show count data"):
        st.subheader("Count data")
        st.dataframe(
            counts.sum(axis=1)
            .sort_values(ascending=False)
            .rename("Number of incidents")
            .to_frame()
            .join(lsoa_boundaries["LSOA21NM"].rename("LSOA name"))
        )

    # st.subheader("Number of pickups by hour")

    # hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]

    # st.bar_chart(hist_values)

    # hour_to_filter = st.slider('hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h
    # filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
    # st.subheader(f'Map of all pickups at {hour_to_filter}:00')

    # TODO play with streamlit-folium https://github.com/randyzwitch/streamlit-folium

    st.map(data, size=100)


if __name__ == "__main__":
    main()
