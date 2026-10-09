# Safer Streets Core

Core Python library for the Safer Streets project: statistical and geospatial primitives for analysing open crime,
geographic and demographic data. This repo is an editable local dependency of the EDA, tooling and apps repos - so you
probably won't need to access this repo directly.

**Assimilation of all the data is now handled by the [safer-streets-tooling](https://github.com/safer-streets/safer-streets-tooling) repo.**

## Setup

See [the project page](https://github.com/safer-streets). For development:

```sh
uv sync --group dev
uv run pytest
```

Configuration is read from environment variables (or the nearest `.env` file) via
[config.py](src/safer_streets_core/config.py). Each setting is only required by the code that uses it:

| Variable | Used for |
| -------- | -------- |
| `SAFER_STREETS_DATA_DIR` | Local data directory |
| `SAFER_STREETS_BLOB_STORAGE` | Azure Blob storage URL |
| `AZURE_STORAGE_CONNSTR` / `AZURE_STORAGE_ADMIN_CONNSTR` | Azure Blob read / write access |
| `NOMIS_API_KEY` | Nomisweb census API |
| `CARTO_API_KEY` | Carto (optional) |

See [doc/azure-blob-access.md](doc/azure-blob-access.md) for how access to the Azure Blob storage is controlled and
which connection string to use.

## Content Overview

Library modules in [src/safer_streets_core/](src/safer_streets_core/):

| Module | Content |
| ------ | ------- |
| [measures.py](src/safer_streets_core/measures.py) | Concentration (Lorenz curves, Gini and Poisson-adjusted Gini, overdispersion) and similarity/stability (Spearman rank correlation, rank-biased overlap, cosine similarity) |
| [stats.py](src/safer_streets_core/stats.py) | Distribution fitting: Poisson, negative binomial, gamma, exponential, lognormal; Poisson-gamma model |
| [spatial.py](src/safer_streets_core/spatial.py) | Spatial units (census geographies, square/hex/H3 grids, street networks, force boundaries) and mapping points to them |
| [database.py](src/safer_streets_core/database.py) | DuckDB connections (spatial/H3 extensions, Azure, MotherDuck), GeoParquet read/write |
| [utils.py](src/safer_streets_core/utils.py) | Police force names, months, crime data loading, pseudo- and quasirandom crime sampling, data source config |
| [nomisweb.py](src/safer_streets_core/nomisweb.py) | Data model and API access for Nomisweb census/demographic data |
| [file_storage.py](src/safer_streets_core/file_storage.py) | Local and Azure Blob storage backends |
| [api_helpers.py](src/safer_streets_core/api_helpers.py) | Generic HTTP/JSON/GeoDataFrame API helpers |
| [charts.py](src/safer_streets_core/charts.py) | Matplotlib helpers (radar charts, map defaults) |
| [models.py](src/safer_streets_core/models.py) | Pydantic models |

Remote data locations (URLs, cached filenames) are in [config/data_sources.json](config/data_sources.json); a copy in
the data directory overrides it. Geometry is in British National Grid (EPSG:27700) unless stated otherwise.

Data download, processing and syncing are handled by
[safer-streets-tooling](https://github.com/safer-streets/safer-streets-tooling).
