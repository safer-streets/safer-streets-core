# Safer Streets Core

Core tooling for the Safer Streets project

Using data from:

- [police.uk](https://data.police.uk/) bulk downloads/API
- ONS for census geographies (e.g. https://geoportal.statistics.gov.uk/datasets/ons::lower-layer-super-output-areas-december-2021-boundaries-ew-bsc-v4-2/about)
- [Nomisweb](https://www.nomisweb.co.uk/) for demographic data (requires an API key)

## Setup

Use a common environment for all safer streets repos (apart from documentation)

```sh
uv venv --python 3.12
uv pip install .
```

Set the env var `NOMIS_API_KEY` with your Nomis API key. Ideally put it in a `.env` file, in the safer-streets root
directory

## Content Overview

### General

- Police & crime metadata
- Data retrieval and man
- Concentration computations: Lorenz, Gini etc
- Similarity computations, e.g rank correlation

### Geopatial

- Spatial units/scales:
  - [X] Administrative: PFAs
  - [X] Census/statistical: MSOA21, LSOA21, OA21
  - [X] Regular: Square, hex, h3 grids
  - [X] Street segments
- Point aggregration to above units
- Clumpiness calculation

### Demographics

- Data model and API access for Nomisweb


