# Safer Streets Core

Core tooling for the Safer Streets project

Using data (mainly) from:

- [police.uk](https://data.police.uk/) bulk downloads/API
- ONS for census geographies (e.g. https://geoportal.statistics.gov.uk/datasets/ons::lower-layer-super-output-areas-december-2021-boundaries-ew-bsc-v4-2/about)
- [Nomisweb](https://www.nomisweb.co.uk/) for demographic data (requires an API key)

## Setup

See [the project page](https://github.com/safer-streets)

## Building the database

`build-db` runs the full data pipeline in one reproducible pass, producing a single
DuckDB database. The pipeline:

1. **Crime data** — downloads the latest police.uk bulk archive and loads it into a
   `crime_data` table (point geometry in British National Grid).
2. **ONS boundaries** — downloads the generalised boundary layers (PFA, LAD, MSOA, LSOA, OA)
   into one table each. Each layer is cached as a GeoPackage under the data directory and
   reused on later runs (pass `--force-download` to refresh).
3. **Supplementary layers** — `open_greenspace` and `open_roads` are downloaded from the OS
   Downloads API (open data, no API key) and cached as zips (reused unless `--force-download`);
   `poi` is streamed from [Overture Maps](https://overturemaps.org/) places (selected categories)
   straight into DuckDB via the `overturemaps` reader (no intermediate file). `land_cover` is
   loaded from a [UKCEH Land Cover Map](https://catalogue.ceh.ac.uk/) vector GeoPackage (licensed
   — see [Manual downloads](#manual-downloads)). Any of these is skipped with a warning if absent.
4. **H3 aggregations** — first repairs invalid geometries (`ST_MakeValid`) and builds an
   RTree spatial index on the geometry tables (all `geom` tables except `crime_data`), then
   builds, for each H3 resolution, `crime_counts_h3_{res}` (counts by cell / crime type /
   month), `h3_{res}_{key}_lookup` views (cell → ONS geography), `h3_{res}_{name}_lookup`
   views (cell → each overlapping greenspace / land-cover polygon or road segment, when
   present), and `h3_{res}_geogs` (one row per cell with every ONS code plus `greenspace_ids`
   / `land_cover_ids` / `road_ids` lists).

To avoid serving a half-built database, the pipeline writes everything to a
`<name>.staging.db` file and only promotes it with an atomic swap once every stage has
succeeded. Read-only consumers therefore always see a complete database — either the
previous one or the new one, never a partial state. A monthly full rebuild is intentional:
it keeps the logic simple and reliable (no incremental/merge state to maintain).

### Configuration

The data directory is read from a `.env` file (resolved relative to where the command is run):

- `SAFER_STREETS_DATA_DIR` — directory for downloaded/output data

The database is written there as `safer_streets.db`. Pass `--db-path` to write elsewhere.

### Running

```sh
# Build the live database named in SAFER_STREETS_DATABASE
build-db

# Override the output location
build-db --db-path /path/to/safer_streets.db

# Force a fresh download of the crime archive (otherwise a cached copy is reused)
build-db --force-download

# Limit H3 resolutions or boundary layers
build-db --resolutions 9 --layers lad --layers lsoa

# Resume an interrupted build: reuse the existing staging database, skip any stage
# whose output table is already present (crime, boundaries, greenspace, land cover,
# roads), and keep existing H3 tables/views. Useful after a failure part-way through
# (e.g. to avoid re-doing the ~1 GB road download).
build-db --no-replace
```

### Manual downloads

Most inputs are fetched automatically. One source is licensed and must be downloaded by hand
into the data directory (`SAFER_STREETS_DATA_DIR`); the build skips it with a warning if it is
not present:

| Dataset | Table | Source | Where to put it |
| --- | --- | --- | --- |
| UKCEH Land Cover Map (vector) | `land_cover` | [EIDC catalogue](https://catalogue.ceh.ac.uk/) — requires (free) registration and licence acceptance | the downloaded zip, named as the `LAND_COVER_ZIP` constant in `build_db.py`, placed directly in the data directory (the build reads the `.gpkg` inside it) |

Everything else — the police.uk crime archive, ONS boundaries, OS Open Greenspace, OS Open
Roads and Overture Maps POI — is open data and fetched automatically by `build-db`.

## Content Overview

### General

- Police & crime metadata
- Data retrieval and management
- Concentration computations: Lorenz, Gini etc
- Similarity/stability computations, e.g rank correlation

### Statistical

- Pseudo- and quasirandom sampling
- Fitting to distributions

### Geopatial

- Spatial units/scales:
  - [X] Administrative: PFAs
  - [X] Census/statistical: MSOA21, LSOA21, OA21
  - [X] Regular: Square, hex, h3 grids
  - [X] Street segments
- Point aggregration to above units
- Clumpiness calculation
- Population redistribution (from census to other spatial units)

### Demographics

- Data model and API access for Nomisweb
