# Agent Guidelines for `safer-streets-core`

This file instructs AI agents acting as developer, reviewer, and QA for this repository.

## Project Overview

`safer-streets-core` is the core Python tooling for the Safer Streets project: it retrieves
open crime, geographic, demographic and deprivation data, and builds a single reproducible
DuckDB database from it. It also provides the statistical and geospatial primitives (crime
concentration measures, distribution fitting, spatial aggregation, population redistribution)
used to analyse that data.

The library source lives in [src/safer_streets_core/](src/safer_streets_core/). Key modules:

| File | Role |
| ---- | ---- |
| [utils.py](src/safer_streets_core/utils.py) | Shared utilities: `data_dir`, `data_source`, `Force`, archive/download helpers |
| [database.py](src/safer_streets_core/database.py) | DuckDB connection context + spatial/vss/h3 extension loading |
| [spatial.py](src/safer_streets_core/spatial.py) | Spatial units (PFA/MSOA/LSOA/OA, hex/H3 grids, street networks), point aggregation, isochrones |
| [measures.py](src/safer_streets_core/measures.py) | Concentration measures: Lorenz, Gini, etc. |
| [stats.py](src/safer_streets_core/stats.py) | Distribution fitting (Poisson, negative binomial, gamma, lognormal, …) |
| [nomisweb.py](src/safer_streets_core/nomisweb.py) | Data model + API access for Nomisweb census/demographic data (needs an API key) |
| [api_helpers.py](src/safer_streets_core/api_helpers.py) | Generic HTTP/JSON/GeoDataFrame API helpers and `ApiError` |
| [file_storage.py](src/safer_streets_core/file_storage.py) | `DataSource` protocol + Azure Blob storage backend |
| [models.py](src/safer_streets_core/models.py) | Pydantic models (e.g. `Neighbourhood`) |
| [charts.py](src/safer_streets_core/charts.py) | Matplotlib chart helpers (radar charts, map defaults) |

CLI entry points (defined under `[project.scripts]` in [pyproject.toml](pyproject.toml)) live in
[src/scripts/](src/scripts/):

| Command | Module | Role |
| ------- | ------ | ---- |
| `extract` | [extract.py](src/scripts/extract.py) | Download and load police.uk street-level crime data |
| `assign-population` | [assign_population.py](src/scripts/assign_population.py) | Redistribute census population to other spatial units |
| `cloud` | [azure_sync.py](src/scripts/azure_sync.py) | Sync data to/from Azure Blob storage |

Other pipeline pieces: [ons_boundaries.py](src/scripts/ons_boundaries.py) (ONS boundary downloads)
and [ew_hex200.py](src/scripts/ew_hex200.py) (200 m hex grid over England & Wales).

Data source locations (URLs, cached filenames, layer hints) live in
[config/data_sources.json](config/data_sources.json), read via `utils.data_source`. Tests are in
[src/test/](src/test/).

## Toolchain

| Tool | Command |
| ---- | ------- |
| Package manager | `uv` |
| Linter / formatter | `ruff` (`uv run ruff check`, `uv run ruff format`) |
| Type checker | `ty` (`uv run ty check`) |
| Tests | `uv run pytest` |
| Install dev deps | `uv sync --group dev` |

Pre-commit hooks run `uv-lock`, `ruff-check --fix`, and `ruff-format` automatically on commit
(`ty` is not yet wired into pre-commit — run it manually). See [.pre-commit-config.yaml](.pre-commit-config.yaml).

## Quality Gates

All of the following must pass before any change is considered complete:

```sh
uv run ruff check          # zero lint errors
uv run ruff format --check # zero formatting issues
uv run ty check            # zero type errors
uv run pytest              # all tests pass, coverage >= 65%
```

`pytest` is configured (in [pyproject.toml](pyproject.toml)) with `--cov=src/safer_streets_core`,
an HTML coverage report, and `--cov-fail-under=65`. Do not let coverage drop below the threshold.

Most tests run without external data: CI sets `SAFER_STREETS_DATA_DIR=""` and the suite passes
without a Nomisweb API key or any downloaded inputs. Keep new tests offline-safe — mock or skip
anything that needs network access, a real data directory, or an API key.

## Developer Rules

- **Runtime vs dev dependencies.** Runtime deps go in `[project.dependencies]`; tooling
  (`ruff`, `ty`, `pytest`, `pre-commit`, `datamodel-code-generator`) goes in
  `[dependency-groups.dev]`. New runtime deps need a strong justification — this is already a
  heavy stack (geopandas, duckdb, osmnx, scikit-learn, azure-*).
- **Downloads are cached.** Stages reuse cached inputs unless `--force-download` is passed.
  Preserve this behaviour when touching download/extract logic.
- **Data source config is externalised.** New remote URLs, cached filenames or layer/sheet hints
  belong in [config/data_sources.json](config/data_sources.json) (read via `utils.data_source`),
  not hard-coded. A copy dropped in the data directory must still override the bundled one.
- **Coordinate reference systems matter.** Geometry is stored in British National Grid (EPSG:27700)
  unless stated otherwise. Be explicit about CRS on any new spatial operation; mixing CRSs silently
  produces wrong results.
- **Type annotations required.** All function signatures need full annotations. `ty` will catch
  missing or incorrect ones. A few `ty` rules are relaxed in [pyproject.toml](pyproject.toml)
  (`invalid-assignment`, `unresolved-attribute`, `no-matching-overload`) — prefer fixing types
  over widening those exemptions, and use targeted `# ty: ignore[...]` comments for genuine edge cases.
- **Line length is 120** (configured in [pyproject.toml](pyproject.toml) under `[tool.ruff]`; `E501` is ignored).
- **No comments explaining what the code does.** Only add a comment when the *why* is non-obvious
  (hidden constraint, workaround, subtle invariant).

## Reviewer Checklist

When reviewing a PR or diff, check:

1. **Correctness** — for spatial/statistical code, reason about edge cases: empty inputs, CRS
   mismatches, invalid geometries (`ST_MakeValid`), missing optional layers.
2. **Offline-safe tests** — new tests must pass with `SAFER_STREETS_DATA_DIR=""` and no API key /
   network. Anything needing external resources must be mocked or skipped.
3. **Coverage** — does the change keep total coverage at or above the 65% gate? New code paths
   (measures, error handling) need tests in [src/test/](src/test/).
4. **CRS correctness** — every new geometry operation should be explicit and consistent about its
   coordinate reference system.
5. **Config, not constants** — new data sources go in [config/data_sources.json](config/data_sources.json).
6. **Types** — precise annotations; avoid `Any` unless unavoidable; don't broaden the relaxed `ty`
   rules to dodge a real type error.
7. **Ruff rules** — no rule in the `select` list should be suppressed without justification.
   Active rules: `A, B, E, F, I, SIM, UP` (`E501` ignored; `D103` also ignored in test files).
8. **Docs** — if the build pipeline, CLI flags, manual-download requirements or public behaviour
   change, update [README.md](README.md).

## QA Rules

- Run the full gate suite (`ruff check`, `ruff format --check`, `ty check`, `pytest`) before
  declaring any task done.
- CI runs the matrix Python 3.13 × ubuntu, macos, windows ([lint-test.yml](.github/workflows/lint-test.yml)).
  Flag anything that might be platform-specific, especially path handling and file-locking around
  DuckDB / the atomic database swap (Windows is stricter about open file handles than Linux/macOS).
- The minimum supported Python is **3.13** (`requires-python = ">=3.13"`). Don't introduce syntax
  or stdlib APIs newer than that.
- If a test is skipped or marked `xfail`, leave a comment explaining why and when it can be removed.

## Repository Layout

```
src/
  safer_streets_core/
    __init__.py        # package version
    utils.py           # shared utilities: data_dir, data_source, Force, downloads
    database.py        # DuckDB connection context + extension loading
    spatial.py         # spatial units, point aggregation, street networks, isochrones
    measures.py        # concentration measures (Lorenz, Gini, ...)
    stats.py           # distribution fitting
    nomisweb.py        # Nomisweb census/demographic API + data model
    api_helpers.py     # generic HTTP/JSON/GeoDataFrame helpers, ApiError
    file_storage.py    # DataSource protocol + Azure Blob backend
    models.py          # pydantic models
    charts.py          # matplotlib chart helpers
    py.typed           # PEP 561 marker
  scripts/
    extract.py         # `extract` — police.uk crime data
    assign_population.py# `assign-population` — population redistribution
    azure_sync.py      # `cloud` — Azure Blob sync
    ons_boundaries.py  # ONS boundary downloads
    ew_hex200.py       # 200m hex grid over England & Wales
  test/                # pytest suite (one test_*.py per module)
config/
  data_sources.json    # remote URLs, cached filenames, layer hints
.github/workflows/
  lint-test.yml        # CI: lint + type check + test matrix + wheel/coverage artifacts
README.md
pyproject.toml
.pre-commit-config.yaml
```

## Branch and Release Policy

- **`main` is branch-protected.** Direct pushes are blocked; all changes go through a pull request
  targeting `main`. CI (all OS combinations) must pass before merging.
- CI on `main` (and tags) builds a wheel and uploads it, plus the HTML coverage report, as
  artifacts ([lint-test.yml](.github/workflows/lint-test.yml)). There is currently no automated
  PyPI publish.
- Version bumps go in [pyproject.toml](pyproject.toml) (`version = "x.y.z"`).

## Workflow

1. Create a feature branch off `main` — never commit directly to `main`.
2. Make changes under [src/safer_streets_core/](src/safer_streets_core/) or [src/scripts/](src/scripts/).
3. Add or update tests in [src/test/](src/test/) — keep them offline-safe and keep coverage at/above 65%.
4. Run the full gate suite locally (`ruff check`, `ruff format --check`, `ty check`, `pytest`).
5. If the build pipeline, CLI flags, manual-download requirements or public behaviour changed,
   update [README.md](README.md) and, if relevant, [config/data_sources.json](config/data_sources.json).
6. Commit — pre-commit hooks will auto-fix formatting and re-lock `uv.lock`.
7. Open a PR targeting `main`; CI must pass across ubuntu/macos/windows before merging.
