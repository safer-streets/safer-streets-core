# Checkpoint July 2025

## Building infra around (public) police and geo data

- Extraction from data.police.uk
- Unified interface for spatial units/spatial querying: statistical geographies, regular lattices, streets
- Interface for demographic data (census 2021 via Nomisweb) OAC etc

## EDA

- [X] Comparison of spatial units
  - [ ] quadtree representation?

Ways of measuring concentration (spatial)

- [X] Lorenz and naive Gini
- [X] Clumpiness index (of top n %)
- [X] Adjusted Gini
  - [ ] gradually cluster random data and demonstrate how Adj Gini changes more than naive.
  - [X] is random data accurately modelled by a Poisson dist? (c.f negative binomial) - yes, with caveats
- [X] KDE
- [X] Poisson-Gamma

Ways of measuring predictivity (temporal variation)

- [X] Rank correlation
- [X] Rank-biased overlap (RBO)
  - [ ] most persistent spatial units in top n %.
- [X] F1 score as overlap
- [X] Cosine similarity
- [X] Similarity scores: 2d Kolmogorov-Smirnov (but there is no definitive distribution?)
- Seasonality
  - [ ] for given areas
  - [X] of concentration as a whole


Interventions - cost/diffusion/half-life/ramp-up/reduction potential (diffusion could influence clumpy areas)

Mohler/Bernasco


## Demographics

- [X] OAC correlation
- [ ] IMD
- [ ] OA-level census (ethnicity, NSSEC...)

## Streamlit app

- [ ] Pickers for crime type, spatial unit, total (% or abs?) area to cover
- [ ] Animate map of top spatial units by crime count over time
- [ ] Graph population impacted by ethnicity / IMD / NSSEC

## Writeup

- [ ] comparison of measures for concentration/predictivity