# Measurement Report Localization Based on Ray Tracing Fingerprinting

Localization of anonymous LTE Measurement Reports using radio fingerprinting. Builds a fingerprint database from ray tracing simulation data, then applies Weighted K-Nearest Neighbors (WKNN) matching to estimate the geographic position of each MR. Outputs coverage quality maps for network optimization.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Approach](#approach)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Results](#results)
- [Limitations and Future Work](#limitations-and-future-work)

---

## Problem Statement

In mobile networks, User Equipment (UE) continuously sends **Measurement Reports (MR)** to the serving base station. Each MR contains the Reference Signal Received Power (RSRP) from the serving cell and neighboring cells — but carries **no GPS coordinates** and **no user identity**.

Given a set of anonymous MRs, the goal is to **estimate the geographic position** of each report. This enables network engineers to map real-world signal quality to physical locations, supporting coverage monitoring and network optimization without requiring drive tests or user tracking.

### Formal definition

| | Description |
|---|---|
| **Input** | A Measurement Report: `{serving_cell_id, serving_rsrp, {neighbor_cell_id: rsrp, ...}}` |
| **Output** | Estimated position `(sim_x, sim_y)` in meters (Cartesian coordinate system) |
| **Method** | Radio fingerprinting — match the observed RSRP vector against a pre-built fingerprint database |

---

## Dataset

### `data/raw/gcell_conf.csv` — Network configuration

Cell-level configuration for all 26 cells across 10 gNodeBs in the study area.

| Column | Type | Description |
|---|---|---|
| `gcell_id` | string (UUID) | Unique cell identifier |
| `gnodeb_id` | string (UUID) | Parent base station ID |
| `sim_x` | float (m) | BTS position — x axis (Cartesian) |
| `sim_y` | float (m) | BTS position — y axis (Cartesian) |
| `antenna_height` | float (m) | Antenna height above ground |
| `azimuth` | float (°) | Sector pointing direction (0° = North) |
| `digital_tilt` | float (°) | Electrical downtilt angle |
| `sync_date` | string | Configuration date |

**Summary:** 10 gNBs · 26 cells · 2–3 sectors per gNB · study area approx. 2.2 km × 1.2 km

### `data/raw/measurement_data.csv` — Simulation measurement data

Ray tracing simulation output: RSRP from each cell at each UE position. Serves as both the **fingerprint database source** and the **ground truth for evaluation**.

| Column | Type | Description |
|---|---|---|
| `ue_id` | string (UUID) | UE identifier |
| `sim_x` | float (m) | UE position — x axis (**ground truth**) |
| `sim_y` | float (m) | UE position — y axis (**ground truth**) |
| `date` | string | Measurement timestamp |
| `rsrp` | float (dBm) | Received signal power from `gcell_id` |
| `gcell_id` | string (UUID) | Cell that was measured |
| `ue_height` | float (m) | UE height above ground (typically 0.5 m) |

**Summary:** 41,481 records · 1,490 UEs · 8,403 unique positions · 1–4 cells measured per timestamp · RSRP range −117 to −43 dBm

> **Note:** This file is in **long format** — each row is one `(UE position, cell, RSRP)` triple. Building the fingerprint database requires pivoting to wide format (one row per position, one column per cell).

> **Data quality:** 6 records with `rsrp = 0` should be filtered before use (simulation artifacts). Records with `rsrp ≥ −50 dBm` are valid — they correspond to UEs located very close to a BTS.

### Coordinate system

All positions use a **local Cartesian coordinate system** (unit: meters, origin arbitrary). There is no GPS/lat-lon conversion needed within this project. The bounding box of the study area:

```
sim_x: −1212 m  to  −145 m
sim_y:  −662 m  to   491 m
```

---

## Approach

### Overview

```
measurement_data.csv
        │
        ▼
┌─────────────────────┐
│  M0 · EDA           │  Understand data distributions, check cell ID
│                     │  overlap, decide fill value for missing cells
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  M1 · Fingerprint   │  Pivot long → wide format
│       Database      │  One row per (sim_x, sim_y), 26 RSRP columns
└────────┬────────────┘  Fill missing cells with −120 dBm
         │
         ▼
┌─────────────────────┐
│  M2 · WKNN          │  Build query vector from each MR
│       Localization  │  Find K nearest fingerprints by Euclidean distance
└────────┬────────────┘  Weighted average of K positions → (est_x, est_y)
         │
         ▼
┌─────────────────────┐
│  M3 · Evaluation    │  Compare (est_x, est_y) vs ground truth (sim_x, sim_y)
│                     │  Compute mean error, CEP50, CEP90, CDF
└────────┬────────────┘  Analyze error by zone and number of measured cells
         │
         ▼
┌─────────────────────┐
│  M4 · Visualization │  Coverage quality map, localization error map,
│                     │  UE density heatmap
└─────────────────────┘
```

### Fingerprint Database

Each unique `(sim_x, sim_y)` position in `measurement_data` becomes one row in the fingerprint DB. The 26 cell RSRP values become features:

```
sim_x    sim_y    C-31c9  C-4452  C-f54c  ...  C-95d9   (26 columns)
─────────────────────────────────────────────────────────────────────
-442.1   -124.3   -49.2   -78.5   -120    ...  -103.1
 819.4    302.1   -82.0   -120    -91.3   ...  -120
 ...
```

Missing cells (not measured at that position) are filled with `−120 dBm` (approximate noise floor). This value is a hyperparameter that can be tuned.

### WKNN Matching

Given a query MR with observed RSRP values, build a query vector of the same 26-dimensional format and find the K most similar fingerprints:

```
distance(query, fingerprint_i) = √Σ (query_j − fp_i_j)²

position_estimate = Σ (w_i × pos_i) / Σ w_i
where w_i = 1 / distance_i   (inverse distance weighting)
```

### Train / Test Split

To avoid data leakage, split is performed **by UE** using `GroupShuffleSplit`:

```python
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(data, groups=data['ue_id']))
```

This ensures that positions seen during training are not used in evaluation — a UE's trajectory does not appear in both sets.

---

## Project Structure

```
mr-localization/
│
├── README.md
├── pyproject.toml             # Project metadata and dependencies (uv)
├── uv.lock                    # Locked dependency versions (commit this)
├── run_pipeline.py            # Run full pipeline with one command
│
├── data/
│   ├── raw/                   # Original input files (not committed to Git)
│   │   ├── gcell_conf.csv
│   │   └── measurement_data.csv
│   └── processed/             # Generated files (not committed to Git)
│       ├── fp_db.parquet      # Fingerprint database (wide format)
│       └── mr_located.parquet # MR records with estimated positions
│
├── notebooks/
│   └── 00_eda.ipynb           # Exploratory data analysis
│
├── src/
│   ├── __init__.py
│   ├── build_fp_db.py         # M1 — build fingerprint database
│   ├── localize_mr.py         # M2 — WKNN localization
│   ├── evaluate.py            # M3 — compute metrics
│   └── visualize.py           # M4 — generate maps and charts
│
├── results/
│   ├── metrics.json           # Evaluation results
│   ├── cdf_error.png          # CDF of localization error
│   ├── error_by_k.png         # Mean error vs K
│   └── maps/
│       ├── coverage_map.html  # Interactive coverage quality map
│       └── error_map.png      # Spatial distribution of error
│
└── tests/
    └── test_localize.py       # Unit tests for core functions
```

---

## Installation

**Requirements:** Python 3.10+ · [uv](https://docs.astral.sh/uv/)

Install `uv` if you haven't already:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/mr-localization.git
cd mr-localization

uv sync
```

`uv sync` reads `pyproject.toml`, creates a virtual environment in `.venv/`, and installs all dependencies from `uv.lock` — no manual `venv` or `pip install` needed.

Place the raw data files in `data/raw/`:

```
data/raw/gcell_conf.csv
data/raw/measurement_data.csv
```

---

## Usage

### Run the full pipeline

```bash
uv run python run_pipeline.py
```

This executes all four modules in order (M1 → M2 → M3 → M4) and writes outputs to `data/processed/` and `results/`.

### Run individual modules

```bash
# M1 — Build fingerprint database
uv run python src/build_fp_db.py

# M2 — Localize measurement reports
uv run python src/localize_mr.py --k 5

# M3 — Evaluate model
uv run python src/evaluate.py

# M4 — Generate visualizations
uv run python src/visualize.py
```

### Run EDA notebook

```bash
uv run jupyter lab notebooks/00_eda.ipynb
```

### Run tests

```bash
uv run pytest tests/ -v
```

### Add a new dependency

```bash
uv add <package>        # add to pyproject.toml and update uv.lock
uv add --dev <package>  # add as dev-only dependency (e.g. pytest, jupyterlab)
```

---

## Evaluation

### Metrics

| Metric | Definition |
|---|---|
| **Mean Error** | Average Euclidean distance between estimated and true position (meters) |
| **Median Error** | 50th percentile of error distribution |
| **CEP50** | Radius containing 50% of estimates (Circular Error Probable) |
| **CEP90** | Radius containing 90% of estimates |

Distance is computed as Euclidean distance in the Cartesian coordinate system:

```python
error_m = sqrt((est_x − true_x)² + (est_y − true_y)²)
```

### Hyperparameter tuning

The main hyperparameter is **K** (number of nearest neighbors). The pipeline evaluates K ∈ {1, 3, 5, 7, 9, 15} and reports mean error for each. A chart of mean error vs K is saved to `results/error_by_k.png`.

Secondary hyperparameter: **fill value** for missing cells (default `−120 dBm`). Can be changed in `src/build_fp_db.py`.

---

## Results

| K | Mean Error (m) | CEP50 (m) | CEP90 (m) |
|---|---|---|---|
| 1 | — | — | — |
| 3 | — | — | — |
| **5** | **—** | **—** | **—** |
| 7 | — | — | — |
| 9 | — | — | — |

> Results will be filled in after running the pipeline. See `results/metrics.json` for the full output.

---

## Limitations and Future Work

### Current limitations

**Sim-to-real gap.** The fingerprint database is built from ray tracing simulation data. In a real deployment, the simulated RSRP values will differ from measured values due to modeling inaccuracies (antenna pattern, material properties, vegetation). This gap is a known source of localization error.

**Missing neighbor problem.** Real MRs typically report only 1–4 cells (constrained by 3GPP `maxCellReport = 8` and event trigger conditions). The current approach fills missing cells with a constant floor value (`−120 dBm`), which introduces bias in the distance computation. See *Future Work* below for better strategies.

**No temporal modeling.** Each MR is localized independently. UE trajectory information (if available) could improve accuracy.

### Future work

**Improved missing cell handling.** Instead of filling with a constant, compute distance only on the dimensions present in the query vector (partial distance matching). Weight results by the number of matched cells.

**Neural embedding.** Replace the raw 26-dimensional RSRP vector with a learned embedding from an Autoencoder or Triplet Network. A Triplet loss trained with geographic distance as the supervision signal could learn a metric space where Euclidean distance directly reflects physical proximity.

**Direct regression.** Train an MLP to map the RSRP vector directly to `(sim_x, sim_y)` coordinates, bypassing KNN entirely. The ground truth coordinates in `measurement_data` make this straightforward to implement and evaluate.

**Calibration with sparse real measurements.** Apply a small number of real drive-test measurements to calibrate the sim-to-real offset, following the approach of Geo2SigMap (Li et al., 2024).

---

## References

- 3GPP TS 36.331 — *Evolved Universal Terrestrial Radio Access (E-UTRA); Radio Resource Control (RRC); Protocol specification*
- Rappaport, T. S. — *Wireless Communications: Principles and Practice* (2nd ed.)
- Li et al. (2024) — *Geo2SigMap: High-Fidelity RF Signal Mapping Using Geographic Databases* · [arXiv:2312.14303](https://arxiv.org/abs/2312.14303)
- Tse & Viswanath — *Fundamentals of Wireless Communication* · [Free PDF](https://web.stanford.edu/~dntse/papers/book121004.pdf)