from time import sleep
from typing import get_args

import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from itrx import Itr
from spatial import get_force_boundary, map_to_spatial_unit
from utils import (
    CATEGORIES,
    Force,
    Month,
    calc_gini,
    load_crime_data,
    monthgen,
)

LATEST_DATE = Month(2025, 5)
all_months = Itr(monthgen(LATEST_DATE, backwards=True)).take(36).rev().collect()


@st.cache_data
def cache_crime_data(force: Force, category: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    force_boundary = get_force_boundary(force)
    data = load_crime_data(force, all_months, filters={"Crime type": category}, keep_lonlat=True)
    return data, force_boundary


st.set_page_config(layout="wide", page_title="Safer Streets", page_icon="👮")

geographies = {
    "Middle layer super output area (census)": ("MSOA", {}),
    "Lower layer super output area (census)": ("LSOA", {}),
    "Output Area (census)": ("OA", {}),
    "Grid 800m": ("GRID", {"size": 800.0}),
    "Grid 400m": ("GRID", {"size": 400.0}),
    "Grid 200m": ("GRID", {"size": 200.0}),
    "Hex 500m": ("HEX", {"size": 500.0}),
    "Hex 250m": ("HEX", {"size": 250.0}),
    "Hex 125m": ("HEX", {"size": 125.0}),
}


def main() -> None:
    raise RuntimeError("DEPRECATED use the implementation in safer-streets-app")
    st.title("Crime concentration explorer")

    cols = st.columns(2)

    force = cols[0].selectbox("Force Area", get_args(Force), index=43)  # default="West Yorkshire"

    category = cols[0].selectbox("Crime type", CATEGORIES, index=1)

    spatial_unit_name = cols[1].selectbox("Spatial Unit", geographies.keys(), index=0)

    top_percent = cols[1].select_slider(
        "(Uptown) Top rankin",
        np.linspace(10, 100, 10),
        value=50,
        help="Focus on the spatial units comprising the desired percentage of total crimes",
    )
    top_frac = 1 - top_percent / 100
    raw_data, boundary = cache_crime_data(force, category)

    centroid_lat, centroid_lon = raw_data.lat.mean(), raw_data.lon.mean()

    spatial_unit, spatial_unit_params = geographies[spatial_unit_name]
    crime_data, features = map_to_spatial_unit(raw_data, boundary, spatial_unit, **spatial_unit_params)

    # now convert everything to Webmercator
    crime_data = crime_data.to_crs(epsg=4326)
    boundary = boundary.to_crs(epsg=4326)
    features = features.to_crs(epsg=4326)

    counts = (
        crime_data.groupby(["Month", "spatial_unit"])["Crime type"]
        .count()
        .unstack(level="Month", fill_value=0)
        .sort_index()
    )
    counts = counts.reindex(features.index, fill_value=0)
    num_features = len(features)

    st.toast("Data loaded")

    show_lorenz_curve = st.checkbox("Show Lorenz curve")

    view_state = pdk.ViewState(
        latitude=centroid_lat,
        longitude=centroid_lon,
        zoom=9,
        pitch=45,
    )

    boundary_layer = pdk.Layer(
        "GeoJsonLayer",
        boundary.__geo_interface__,
        opacity=0.5,
        stroked=True,
        filled=True,
        get_fill_color=[0, 0, 200, 80],  # 180, 0, 200, 80
        get_line_color=[255, 255, 255, 255],
    )

    def render(m: str, c: pd.Series) -> None:
        lorenz = c.sort_values().cumsum() / c.sum()
        top_features = features[["geometry"]].loc[lorenz[lorenz >= top_frac].index].join(c.rename("n_crimes"))

        gini, lorenz_pct = calc_gini(c)

        title.markdown(f"""
            ## {m}

            {len(top_features) / num_features:.1%} ({len(top_features)}/{num_features}) of spatial units contain
            {top_percent}% of crime

            **Gini Coefficient = {gini:.2f}**

            """)
        if show_lorenz_curve:
            lorenz_pct = lorenz_pct.to_frame()
            lorenz_pct["thresh"] = top_percent / 100
            graph.line_chart(lorenz_pct, x_label="Proportion of spatial units", y_label="Proportion of crime")

        feature_layer = pdk.Layer(
            "GeoJsonLayer",
            top_features.__geo_interface__,
            opacity=1,
            stroked=True,
            filled=True,
            extruded=True,
            wireframe=True,
            get_fill_color=[255, 0, 0, 160],
            get_line_color=[255, 255, 255, 255],
            # pickable=True,
            elevation_scale=20,
            get_elevation="properties.n_crimes",
        )
        map_placeholder.pydeck_chart(pdk.Deck(layers=[boundary_layer, feature_layer], initial_view_state=view_state))

    def render_static():
        m = str(st.session_state.month_slider)
        if m not in counts.columns:
            st.error(f"No data for {m}")
        else:
            render(m, counts[m])

    def render_dynamic():
        for m, c in counts.items():
            render(m, c)
            sleep(0.5)

    cols = st.columns(2)
    run_button = cols[0].empty()
    month_select = cols[1].empty()

    title = st.empty()
    map_placeholder = st.empty()
    map_placeholder.pydeck_chart(pdk.Deck(layers=[boundary_layer], initial_view_state=view_state))

    graph = st.empty()

    run = run_button.button("Run all...")
    z = month_select.select_slider("...or pick a month", options=all_months, value=all_months[0], key="month_slider")

    if run:
        render_dynamic()
    if z:
        render_static()


if __name__ == "__main__":
    main()
