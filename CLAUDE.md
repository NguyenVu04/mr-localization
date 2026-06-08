# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MR Localization via Radio Fingerprinting. Estimates the geographic position of anonymous LTE Measurement Reports (MRs) by matching observed RSRP vectors against a pre-built fingerprint database using Weighted K-Nearest Neighbors (WKNN). Input data comes from ray tracing simulation (not real drive tests).

## Commands

```bash
# Install dependencies (creates .venv/ automatically)
uv sync

# Run the full pipeline (M1 → M2 → M3 → M4)
uv run python run_pipeline.py

# Run individual modules
uv run python src/build_fp_db.py
uv run python src/localize_mr.py --k 5
uv run python src/evaluate.py
uv run python src/visualize.py

# EDA notebook
uv run jupyter lab notebooks/

# Tests
uv run pytest tests/ -v

# Add dependency
uv add <package>
uv add --dev <package>   # dev-only (pytest, jupyterlab, etc.)
```

## Architecture

The pipeline is linear: M1 builds the fingerprint DB from `data/raw/`, M2 runs WKNN localization, M3 evaluates against ground truth, M4 generates charts. All intermediate artifacts are saved to `data/processed/` and `results/`.

```
data/raw/measurement_data.csv  →  M1 (src/build_fp_db.py)   →  data/processed/fp_db.parquet
                                   M2 (src/localize_mr.py)   →  data/processed/mr_located.parquet
                                   M3 (src/evaluate.py)      →  results/metrics.json
                                   M4 (src/visualize.py)     →  results/*.png, results/maps/
```

## Key Data Concepts

**`measurement_data.csv` is long-format.** Each row is one `(UE position, cell, RSRP)` triple. Building the fingerprint DB requires pivoting: group by `(sim_x, sim_y)`, unstack `gcell_id` as columns, aggregate RSRP with `mean`. Missing cells fill to `−120.0` dBm (noise floor constant).

**Query simulation:** group by `(ue_id, date)` to reconstruct a single measurement event — the analog of one real MR.

**Train/test split is by UE** (not by row) using `GroupShuffleSplit(test_size=0.2, random_state=42, groups=ue_id)`. This prevents leakage where positions from the same UE appear in both sets.

**Data quality:** filter 6 records where `rsrp == 0` (simulation artifacts) before any processing.

## Module Responsibilities

| Module | Input | Output | Key logic |
|---|---|---|---|
| `src/build_fp_db.py` | `measurement_data.csv`, `gcell_conf.csv` | `fp_db.parquet` | pivot + fillna(-120) + save |
| `src/localize_mr.py` | `fp_db.parquet`, query records | `mr_located.parquet` | `KNeighborsRegressor(weights='distance')`, K as CLI arg |
| `src/evaluate.py` | `mr_located.parquet` | `results/metrics.json` | Euclidean error, CEP50/90, RMSE, sweep K ∈ {1,3,5,7,9,15} |
| `src/visualize.py` | `mr_located.parquet`, `gcell_conf.csv` | PNG/HTML charts | coverage map, error map, CDF, error-vs-K |

## Coordinate System

All positions are in a **local Cartesian system** (meters, no GPS). Study area: `sim_x ∈ [−1212, −145]`, `sim_y ∈ [−662, 491]`. Never convert to lat/lon within this project.

## Evaluation Metrics

Localization error = Euclidean distance in meters between estimated and true position. Report: Mean Error, Median Error, CEP50, CEP90, RMSE. Primary hyperparameter: K (neighbors). Secondary: fill value for missing cells.
