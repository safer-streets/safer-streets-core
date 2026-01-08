import json
import os
from io import StringIO
from typing import Any

import geopandas as gpd
import pandas as pd
import requests

# by default uses env
URL = os.environ["SAFER_STREETS_API_URL"]
HEADERS = {"x-api-key": os.getenv("SAFER_STREETS_API_KEY")}


def get(endpoint: str, *, url: str | None = None, params: dict[str, Any] | None = None) -> Any:
    """Returns the raw json from a get request"""
    response = requests.get(f"{url or URL}/{endpoint}", params=params, headers=HEADERS)
    response.raise_for_status()
    return response.json()


def post(endpoint: str, payload: Any, *, url: str | None = None) -> Any:
    """Returns the raw json from a post request"""
    response = requests.post(f"{url or URL}/{endpoint}", json=payload, headers=HEADERS)
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
