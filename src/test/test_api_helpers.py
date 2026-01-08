import json
import os
import types

import geopandas as gpd
import pandas as pd

os.environ["SAFER_STREETS_API_URL"] = ""
import safer_streets_core.api_helpers as api_helpers


class DummyResponse:
    def __init__(self, data):
        self._data = data
        self.raise_called = False

    def raise_for_status(self):
        self.raise_called = True

    def json(self):
        return self._data


def test_get_calls_requests_get_and_returns_json(monkeypatch):
    received = {}

    def fake_get(url, params=None, headers=None):
        received["url"] = url
        received["params"] = params
        received["headers"] = headers
        return DummyResponse({"ok": True})

    monkeypatch.setattr(api_helpers, "requests", types.SimpleNamespace(get=fake_get))
    result = api_helpers.get("endpoint", url="http://example.com", params={"q": 1})
    assert result == {"ok": True}
    assert received["url"] == "http://example.com/endpoint"
    assert received["params"] == {"q": 1}
    assert received["headers"] == api_helpers.HEADERS


def test_post_calls_requests_post_and_returns_json(monkeypatch):
    received = {}

    def fake_post(url, json=None, headers=None):
        received["url"] = url
        received["json"] = json
        received["headers"] = headers
        return DummyResponse({"created": True})

    monkeypatch.setattr(api_helpers, "requests", types.SimpleNamespace(post=fake_post))
    payload = {"x": 2}
    result = api_helpers.post("items", payload, url="http://api.local")
    assert result == {"created": True}
    assert received["url"] == "http://api.local/items"
    assert received["json"] == payload
    assert received["headers"] == api_helpers.HEADERS


def test_fetch_df_uses_get_by_default(monkeypatch):
    def fake_get(*args, **kwargs):
        return [{"a": 1}, {"a": 2}]

    monkeypatch.setattr(api_helpers, "get", fake_get)
    df = api_helpers.fetch_df("whatever")
    assert isinstance(df, pd.DataFrame)
    assert df.to_dict("records") == [{"a": 1}, {"a": 2}]


def test_fetch_df_with_http_post_uses_post(monkeypatch):
    def fake_post(*args, **kwargs):
        return [{"b": 3}]

    monkeypatch.setattr(api_helpers, "post", fake_post)
    df = api_helpers.fetch_df("whatever", http_post=True)
    assert isinstance(df, pd.DataFrame)
    assert df.to_dict("records") == [{"b": 3}]


def test_fetch_gdf_uses_get_and_calls_gpd_read_file(monkeypatch):
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"val": 1},
                "geometry": {"type": "Point", "coordinates": [0, 0]},
            }
        ],
    }

    def fake_get(*args, **kwargs):
        return geojson

    captured = {}

    def fake_read_file(file_like):
        # read back the passed content to verify
        captured["content"] = file_like.getvalue()
        return gpd.GeoDataFrame({"val": [1]})

    monkeypatch.setattr(api_helpers, "get", fake_get)
    monkeypatch.setattr(
        api_helpers, "gpd", types.SimpleNamespace(read_file=fake_read_file, GeoDataFrame=gpd.GeoDataFrame)
    )
    gdf = api_helpers.fetch_gdf("geo")
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert json.loads(captured["content"]) == geojson


def test_fetch_gdf_with_http_post_uses_post(monkeypatch):
    geojson = {"type": "FeatureCollection", "features": []}

    def fake_post(*args, **kwargs):
        return geojson

    captured = {}

    def fake_read_file(file_like):
        captured["content"] = file_like.getvalue()
        return gpd.GeoDataFrame()

    monkeypatch.setattr(api_helpers, "post", fake_post)
    monkeypatch.setattr(
        api_helpers, "gpd", types.SimpleNamespace(read_file=fake_read_file, GeoDataFrame=gpd.GeoDataFrame)
    )
    gdf = api_helpers.fetch_gdf("geo", http_post=True)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert json.loads(captured["content"]) == geojson
