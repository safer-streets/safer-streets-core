# Safer Streets Core

Core tooling for the Safer Streets project

Using data (mainly) from:

- [police.uk](https://data.police.uk/) bulk downloads/API
- ONS for census geographies (e.g. https://geoportal.statistics.gov.uk/datasets/ons::lower-layer-super-output-areas-december-2021-boundaries-ew-bsc-v4-2/about)
- [Nomisweb](https://www.nomisweb.co.uk/) for demographic data (requires an API key)

## Setup

See [the project page](https://github.com/safer-streets)

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
