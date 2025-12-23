from unittest.mock import MagicMock, patch

import pytest

from safer_streets_core.nomisweb import (
    api_key,
    build_geog_query,
    fetch,
    fetch_table,
    resolve_table_name,
)


def test_single_code() -> None:
    assert build_geog_query([1]) == "1"


def test_multiple_sequential_codes() -> None:
    assert build_geog_query([1, 2, 3, 4, 5]) == "1...5"


def test_multiple_non_sequential_codes() -> None:
    assert build_geog_query([1, 3, 5]) == "1,3,5"


def test_mixed_sequential_and_non_sequential() -> None:
    assert build_geog_query([1, 2, 3, 5, 6, 7]) == "1...3,5...7"


def test_unsorted_codes() -> None:
    assert build_geog_query([5, 1, 3, 2, 4]) == "1...5"


@patch.dict("os.environ", {"NOMIS_API_KEY": "test_key_123"})
def test_api_key_returns_dict() -> None:
    api_key.cache_clear()
    result = api_key()
    assert result == {"uid": "test_key_123"}


@patch.dict("os.environ", {"NOMIS_API_KEY": "test_key_123"})
def test_api_key_caching() -> None:
    api_key.cache_clear()
    result1 = api_key()
    result2 = api_key()
    assert result1 is result2


@patch("safer_streets_core.nomisweb.requests.get")
@patch.dict("os.environ", {"NOMIS_API_KEY": "test_key"})
def test_fetch_success(mock_get) -> None:
    api_key.cache_clear()
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": "test"}
    mock_get.return_value = mock_response

    result = fetch("test_endpoint", param1="value1")
    assert result == {"data": "test"}
    mock_get.assert_called_once()


@patch("safer_streets_core.nomisweb.requests.get")
@patch.dict("os.environ", {"NOMIS_API_KEY": "test_key"})
def test_fetch_raises_on_error(mock_get) -> None:
    api_key.cache_clear()
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = RuntimeError("HTTP Error")
    mock_get.return_value = mock_response

    with pytest.raises(RuntimeError):
        fetch("test_endpoint")


@patch("safer_streets_core.nomisweb.pd.read_csv")
@patch("safer_streets_core.nomisweb.pd.read_parquet")
@patch("safer_streets_core.nomisweb.Path.exists")
@patch.dict("os.environ", {"NOMIS_API_KEY": "test_key"})
def test_fetch_table_creates_parquet(mock_exists, mock_read_parquet, mock_read_csv) -> None:
    api_key.cache_clear()
    mock_exists.return_value = False
    mock_df = MagicMock()
    mock_read_csv.return_value = mock_df

    fetch_table("test_table")
    mock_read_csv.assert_called_once()
    mock_df.to_parquet.assert_called_once()


@patch("safer_streets_core.nomisweb.pd.read_parquet")
@patch("safer_streets_core.nomisweb.Path.exists")
@patch.dict("os.environ", {"NOMIS_API_KEY": "test_key"})
def test_fetch_table_reads_existing_parquet(mock_exists, mock_read_parquet) -> None:
    api_key.cache_clear()
    mock_exists.return_value = True
    mock_df = MagicMock()
    mock_read_parquet.return_value = mock_df

    result = fetch_table("test_table")
    mock_read_parquet.assert_called_once()
    assert result is mock_df


@patch("safer_streets_core.nomisweb.fetch")
def test_resolve_table_name_single_match(mock_fetch) -> None:
    mock_fetch.return_value = {
        "structure": {
            "keyfamilies": {
                "keyfamily": [
                    {
                        "name": {"value": "Census Table", "lang": "en"},
                        "id": "table_123",
                        "agencyid": "agency",
                        "annotations": {"annotation": []},
                        "components": {
                            "attribute": [],
                            "dimension": [],
                            "primarymeasure": {"conceptref": "test"},
                            "timedimension": {"codelist": "cl", "conceptref": "ct"},
                        },
                        "uri": "http://example.com",
                        "version": 1.0,
                    }
                ]
            },
            "header": {
                "id": "h1",
                "prepared": "2025-01-01",
                "sender": {
                    "id": "s1",
                    "contact": {"email": "test@example.com", "name": "Test", "uri": "http://example.com"},
                },
                "test": "false",
            },
            "xmlns": "http://example.com",
            "common": "http://example.com",
            "structure": "http://example.com",
            "xsi": "http://example.com",
            "schemalocation": "http://example.com",
        }
    }
    result = resolve_table_name("Census")
    assert result == "table_123"


@patch("safer_streets_core.nomisweb.fetch")
def test_resolve_table_name_no_match(mock_fetch) -> None:
    mock_fetch.return_value = {
        "structure": {
            "keyfamilies": None,
            "header": {
                "id": "h1",
                "prepared": "2025-01-01",
                "sender": {
                    "id": "s1",
                    "contact": {"email": "test@example.com", "name": "Test", "uri": "http://example.com"},
                },
                "test": "false",
            },
            "xmlns": "http://example.com",
            "common": "http://example.com",
            "structure": "http://example.com",
            "xsi": "http://example.com",
            "schemalocation": "http://example.com",
        }
    }
    with pytest.raises(ValueError, match="No table found"):
        resolve_table_name("NonExistent")
