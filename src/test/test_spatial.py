from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from safer_streets_core.spatial import (
    _add_centroids,
    get_demographics,
    get_hex_grid,
    get_square_grid,
    load_population_data,
    normalised_clumpiness,
    snap_to_street_segment,
)


class TestAddCentroids:
    def test_add_centroids_creates_required_columns(self):
        """Test that _add_centroids adds BNG_E, BNG_N, LAT, LONG columns"""
        gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs="EPSG:27700")
        result = _add_centroids(gdf)
        assert "BNG_E" in result.columns
        assert "BNG_N" in result.columns
        assert "LAT" in result.columns
        assert "LONG" in result.columns

    def test_add_centroids_values(self):
        """Test that centroid coordinates are calculated correctly"""
        gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])], crs="EPSG:27700")
        result = _add_centroids(gdf)
        assert result.loc[0, "BNG_E"] == 1.0
        assert result.loc[0, "BNG_N"] == 1.0


class TestGetSquareGrid:
    def test_square_grid_creation(self):
        """Test that square grid creates polygons"""
        boundary = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])], crs="EPSG:27700")
        grid = get_square_grid(boundary, size=10, trim=False)
        assert len(grid) > 0
        assert all(isinstance(geom, Polygon) for geom in grid.geometry)

    def test_square_grid_with_offset(self):
        """Test square grid with offset parameter"""
        boundary = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])], crs="EPSG:27700")
        grid = get_square_grid(boundary, size=10, offset=(5, 5), trim=False)
        assert len(grid) > 0

    def test_square_grid_invalid_offset(self):
        """Test that invalid offset raises AssertionError"""
        boundary = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])], crs="EPSG:27700")
        with pytest.raises(AssertionError):
            get_square_grid(boundary, size=10, offset=(20, 20), trim=False)


class TestGetHexGrid:
    def test_hex_grid_creation(self):
        """Test that hex grid creates polygons"""
        boundary = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])], crs="EPSG:27700")
        grid = get_hex_grid(boundary, size=10, trim=False)
        assert len(grid) > 0
        assert all(isinstance(geom, Polygon) for geom in grid.geometry)

    def test_hex_grid_has_centroids(self):
        """Test that hex grid includes centroid columns"""
        boundary = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])], crs="EPSG:27700")
        grid = get_hex_grid(boundary, size=10, trim=False)
        assert "BNG_E" in grid.columns
        assert "BNG_N" in grid.columns


class TestSnapToStreetSegment:
    def test_snap_to_street_segment_adds_columns(self):
        """Test that snap_to_street_segment adds street_segment and distance columns"""
        points = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(10, 10)], crs="EPSG:27700")
        streets = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])], crs="EPSG:27700")
        result = snap_to_street_segment(points, streets)
        assert "street_segment" in result.columns
        assert "distance" in result.columns
        assert len(result) == len(points)


class TestNormalisedClumpiness:
    def test_clumpiness_invalid_scale_zero(self):
        """Test that scale <= 0 raises ValueError"""
        features = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])], crs="EPSG:27700")
        with pytest.raises(ValueError):
            normalised_clumpiness(features, scale=0.0)

    def test_clumpiness_scale_too_large(self):
        """Test that scale larger than bounds raises ValueError"""
        features = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])], crs="EPSG:27700")
        with pytest.raises(ValueError):
            normalised_clumpiness(features, scale=1000.0)

    def test_clumpiness_valid_scale(self):
        """Test clumpiness calculation with valid scale"""
        features = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])], crs="EPSG:27700")
        result = normalised_clumpiness(features, scale=10.0)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestLoadPopulationData:
    @patch("safer_streets_core.spatial.data_dir")
    @patch("safer_streets_core.spatial.tokenize_force_name")
    def test_load_population_data_file_not_found(self, mock_tokenize, mock_data_dir, tmp_path):
        """Test that FileNotFoundError is raised when file doesn't exist"""
        mock_data_dir.return_value = tmp_path
        mock_tokenize.return_value = "nonexistent_force"
        with pytest.raises(FileNotFoundError):
            load_population_data("Nonexistent Force")


class TestGetDemographics:
    def test_get_demographics_returns_dataframe(self):
        """Test that get_demographics returns a DataFrame"""
        population = gpd.GeoDataFrame(
            {
                "geometry": [Point(0, 0), Point(10, 10)],
                "C2021_ETH_20_NAME": ["White", "Asian"],
                "C2021_AGE_6_NAME": ["0-15", "16-64"],
                "C_SEX_NAME": ["Male", "Female"],
            },
            crs="EPSG:27700",
        )
        features = gpd.GeoDataFrame(
            {"geometry": [Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])]},
            index=pd.Index([0], name="spatial_unit"),
            crs="EPSG:27700",
        )
        result = get_demographics(population, features)
        assert isinstance(result, pd.DataFrame)
        assert "count" in result.columns
