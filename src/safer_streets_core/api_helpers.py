import json
import os
from io import StringIO
from typing import Any

import geopandas as gpd
import pandas as pd
import requests


# by default uses env
def default_url() -> str:
    return os.environ["SAFER_STREETS_API_URL"]


def headers() -> dict[str, str | None]:
    return {"x-api-key": os.getenv("SAFER_STREETS_API_KEY")}


def get(endpoint: str, *, url: str | None = None, params: dict[str, Any] | None = None) -> Any:
    """Returns the raw json from a get request"""
    response = requests.get(f"{url or default_url()}/{endpoint}", params=params, headers=headers())
    response.raise_for_status()
    return response.json()


def post(endpoint: str, payload: Any, *, url: str | None = None) -> Any:
    """Returns the raw json from a post request"""
    response = requests.post(f"{url or default_url()}/{endpoint}", json=payload, headers=headers())
    response.raise_for_status()
    return response.json()


def fetch_df(*args: Any, http_post: bool = False, **kwargs: Any) -> pd.DataFrame:
    """Wrap get/post converting (geo)json to DataFrame"""
    method = post if http_post else get
    return pd.DataFrame(method(*args, **kwargs))


def fetch_gdf(*args: Any, http_post: bool = False, **kwargs: Any) -> gpd.GeoDataFrame:
    """Wrap get/post converting (geo)json to GeoDataFrame"""
    method = post if http_post else get
    return gpd.read_file(StringIO(json.dumps(method(*args, **kwargs))))
