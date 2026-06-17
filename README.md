# Safer Streets Core

Core tooling for the Safer Streets project. This repo is an editable local dependency of the EDA, tooling and apps
repos - so you probably won't need to access this repo directly.

**Assimilation of all the data is now handled by the [safer-streets-tooling](https://github.com/safer-streets/safer-streets-tooling) repo.**

(The extract tool - which pulls the public crime data and writes it to parquet - and the cloud tool - which syncs local data with Azure - are retained, but deprecated)

## Setup

See [the project page](https://github.com/safer-streets)

## Content Overview

### General

- Concentration computations: Lorenz, Gini etc
- Similarity/stability computations, e.g rank correlation

### Statistical

- Pseudo- and quasirandom sampling
- Fitting to distributions

### Demographics

- Data model and API access for Nomisweb
