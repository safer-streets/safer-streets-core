# Safer Streets EDA

A place for prototyping and exploration. Not intended to be polished or complete in any way, and subject to change without notice

Using data from:

- [police.uk](https://data.police.uk/) bulk downloads/API
- ONS for census geographies (e.g. https://geoportal.statistics.gov.uk/datasets/ons::lower-layer-super-output-areas-december-2021-boundaries-ew-bsc-v4-2/about)

## Setup

This repo is *not* a package. Create a venv and install from `requirements.txt`. [uv](https://docs.astral.sh/uv/) is highly recommended for managing the environment:

```sh
uv venv --python 3.12
uv pip install -r requirements.txt
```

## Content

Primarily jupyter notebooks:

-[X] `police-api.ipynb`: general exploration and visualisation of the police data for West Yorkshire
-[X] `kde-sampling.ipynb`: spatial crime sampling using KDE
-[X] `poisson-gamma.ipynb`: negative binomial approach in Mohler (2019)
-[ ] `network.ipynb`: sampling on a street network

