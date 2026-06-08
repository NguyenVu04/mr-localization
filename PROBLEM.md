# Problem Statement: MR Localization via Radio Fingerprinting

---

## 1. Background and Motivation

### How measurement reports originate

In LTE and 5G NR networks, User Equipment (UE) continuously measures the radio signal quality of the serving cell and surrounding neighbor cells. When predefined trigger conditions are met — most commonly **Event A3** (neighbor RSRP exceeds serving RSRP by a configurable offset) — the UE sends a **Measurement Report (MR)** to the serving eNB/gNB via the RRC protocol.

Each MR contains:
- The **Physical Cell Identity (PCI)** or **Cell Global Identity (CGI)** of the serving cell
- The **RSRP** (Reference Signal Received Power, in dBm) of the serving cell
- A list of up to `maxCellReport` (typically 6–8) neighbor cells with their respective RSRPs

Crucially, MRs carry **no GPS coordinates** and **no user identity** — they are fully anonymous by design. From any individual MR, the network only knows that *some* device measured *some* signal strengths at *some* location, without knowing which device or where.

### Why localizing MRs matters

The operator's network management system (OMC/EMS) collects and stores hundreds of millions of MRs continuously. If the geographic position of each MR can be estimated, this data becomes a dense, real-time signal quality map — enabling:

- **Coverage monitoring:** identify weak signal zones without drive tests
- **Interference analysis:** locate areas with poor RSRQ despite adequate RSRP
- **Network optimization:** provide spatial input for antenna tilt/power adjustments
- **Self-Organizing Network (SON):** feed localized MRs into automated optimization loops

The key challenge is that MRs lack coordinates, so their positions must be inferred entirely from signal strength patterns.

---

## 2. Core Concept: Radio Fingerprinting

### The fingerprinting intuition

Every geographic location has a unique combination of RSRP values from surrounding cells — a **radio fingerprint**. This uniqueness arises because:

- Each cell is at a different distance from the location
- Each cell points in a different azimuth direction
- Buildings, terrain, and vegetation create location-specific multipath and shadowing patterns

Two locations that are close to the same BTS can have very different fingerprints if they are in different directions relative to the sector antenna, or if one is shielded by a building.

```
Location A (open street, facing cell C1):
  fingerprint = [-49, -82, -103, -91, -120, -78, ...]
                  C1   C2    C3   C4    C5   C6  ...

Location B (behind building, same BTS):
  fingerprint = [-91, -78,  -85, -93,  -68, -82, ...]
                  C1   C2    C3   C4    C5   C6  ...
```

Despite being near the same BTS, the two fingerprints are distinct — because building geometry reshapes the signal field around each specific location.

### The two-phase approach

**Offline phase — build the fingerprint database:**
For every point on a spatial grid covering the study area, record the RSRP from each cell. This produces a lookup table:

```
(sim_x, sim_y) → [rsrp_C1, rsrp_C2, ..., rsrp_C26]
```

In production systems, this database is built from drive test measurements. In this project, it is built from **ray tracing simulation data**, which eliminates the need for field collection.

**Online phase — localize a new MR:**
Given an incoming MR with a set of observed RSRPs, find the fingerprint in the database that most closely matches the observed values. The coordinates of that fingerprint are returned as the estimated position.

---

## 3. Problem Formulation

### Input

A single measurement record consisting of:

| Field | Description |
|---|---|
| `gcell_id` | Cell identifier (one or more cells measured simultaneously) |
| `rsrp` | Received signal power from that cell (dBm) |

Multiple records sharing the same `(ue_id, date)` represent a single measurement event — one UE measuring multiple cells at the same time and position. Grouped together, they form the **query fingerprint vector**:

```python
query = {
    "296382a9...": -82.0,   # cell 1
    "501e096c...": -86.0,   # cell 2
    # cells not measured → filled with -120 dBm (noise floor)
}
```

### Output

Estimated position in the local Cartesian coordinate system:

```
(est_sim_x, est_sim_y)   # unit: meters
```

### Objective

Minimize the **mean localization error** over all test positions:

```
error_i = √( (est_x_i − true_x_i)² + (est_y_i − true_y_i)² )

minimize: E[error_i]   over all i
```

---

## 4. Dataset

### Study area

The study area is an urban environment approximately **1.07 km × 1.15 km** (≈ 1.23 km²), represented in a local Cartesian coordinate system centered at an arbitrary origin:

```
sim_x range:  −1212 m  to  −145 m
sim_y range:   −662 m  to   491 m
```

### File 1: `data/raw/gcell_conf.csv` — Cell configuration

| Column | Description |
|---|---|
| `gcell_id` | Unique cell identifier (UUID) |
| `gnodeb_id` | Parent gNodeB identifier (UUID) |
| `sim_x`, `sim_y` | BTS antenna position (meters, Cartesian) |
| `antenna_height` | Antenna height above ground (m) |
| `azimuth` | Sector boresight direction (degrees, 0° = North) |
| `digital_tilt` | Electrical downtilt (degrees) |
| `sync_date` | Configuration snapshot date |

**Network topology:** 10 gNodeBs · 26 cells · 2–3 sectors per gNB

This file provides the **spatial anchor** for each cell — required both for ray tracing and for contextual analysis of localization results (e.g., error vs. distance to nearest BTS).

### File 2: `data/raw/measurement_data.csv` — Ray tracing output

| Column | Description |
|---|---|
| `ue_id` | UE identifier (UUID) |
| `sim_x`, `sim_y` | **Ground truth UE position** (meters, Cartesian) |
| `date` | Measurement timestamp |
| `rsrp` | Simulated RSRP from `gcell_id` at this UE position (dBm) |
| `gcell_id` | Cell being measured |
| `ue_height` | UE height above ground (m, typically 0.5) |

**Dataset statistics:**

| Metric | Value |
|---|---|
| Total records | 41,481 |
| Unique UEs | 1,490 |
| Unique grid positions | 8,403 |
| RSRP range | −117 to −42 dBm |
| Average cells per position | 2.4 |
| Positions with only 1 cell measured | 3,527 (42%) |
| Positions with 4+ cells measured | 2,108 (25%) |

### Important: format and data nature

This file is in **long format** — each row is one `(UE position, cell, RSRP)` triple, not a traditional MR message. There is **no serving/neighbor cell distinction** — all measured cells are treated equally. This is raw ray tracing output, not a filtered MR log from a real network.

Implications:
- **Building the fingerprint DB:** group by `(sim_x, sim_y)`, pivot `gcell_id` into columns, aggregate RSRP with `mean` (multiple UEs may measure the same position)
- **Simulating the query:** group records by `(ue_id, date)` to reconstruct a single measurement event (1 UE, 1 timestamp, N cells) — this is the analog of one MR
- **Upper bound note:** because there is no `maxCellReport` limit or event trigger filtering, query vectors here are richer than real-world MRs. Localization accuracy will be optimistic compared to a real deployment

### Data quality notes

| Issue | Count | Action |
|---|---|---|
| Records with `rsrp = 0` | 6 | Filter out — simulation artifact |
| Records with `rsrp ≥ −50 dBm` | 23 | Keep — valid, UE is very close to BTS |
| Positions measured by multiple UEs | 464 | Aggregate with `mean` when building fingerprint DB |

---

## 5. Technical Approach

### Module overview

```
measurement_data.csv  ──►  M0: EDA
                               │
                               ▼
                           M1: Build Fingerprint Database
                           (pivot + fill + KD-tree index)
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
               M2: Localize          M3: Evaluate
               (WKNN matching)       (on 20% test split)
                    │                     │
                    └──────────┬──────────┘
                               ▼
                           M4: Visualize
                           (maps + charts)
```

### M0 — Exploratory Data Analysis

Goals: understand data distributions, validate assumptions, document design decisions.

Key questions to answer:
- What is the RSRP distribution per cell? Are there outliers?
- How many cells are visible per position? What is the spatial coverage pattern?
- Do all `gcell_id` values in `measurement_data` appear in `gcell_conf`? (They should — verify this)
- What is the spatial density of grid positions? Are they uniformly distributed?

### M1 — Build Fingerprint Database

**Step 1: Filter** records with `rsrp = 0`.

**Step 2: Pivot** from long to wide format:

```python
fp_db = (
    measurement_data
    .groupby(['sim_x', 'sim_y', 'gcell_id'])['rsrp']
    .mean()                           # aggregate multiple UEs at same position
    .unstack('gcell_id')              # cells become columns
    .reset_index()
    .fillna(-120.0)                   # fill unobserved cells with noise floor
)
# shape: (8403, 28)  →  2 coordinate cols + 26 cell cols
```

**Step 3: Build spatial index** for fast nearest-neighbor lookup:

```python
from scipy.spatial import KDTree
coords = fp_db[['sim_x', 'sim_y']].values
tree = KDTree(coords)
```

**Step 4: Save** to Parquet for fast reload.

**Design decisions to document:**
- Fill value for missing cells: `−120 dBm` (approximate noise floor). Rationale: cells not measured at a position are assumed to be below the sensitivity threshold.
- Aggregation function: `mean`. Rationale: multiple UEs measuring the same position provide independent estimates of the same ground truth signal.

### M2 — WKNN Localization

For each query (one UE at one timestamp), build a 26-dimensional feature vector matching the fingerprint DB column order:

```python
def build_query_vector(records, cell_columns, fill=-120.0):
    vec = np.full(len(cell_columns), fill)
    for _, row in records.iterrows():
        if row['gcell_id'] in cell_index:
            vec[cell_index[row['gcell_id']]] = row['rsrp']
    return vec
```

Find the K nearest fingerprints by Euclidean distance and compute a weighted position estimate:

```
distance_i = √Σ (query_j − fp_i_j)²

weight_i = 1 / distance_i

est_x = Σ (weight_i × fp_i_x) / Σ weight_i
est_y = Σ (weight_i × fp_i_y) / Σ weight_i
```

**Implementation:** use `sklearn.neighbors.KNeighborsRegressor` with `weights='distance'`.

### M3 — Evaluation

**Train/test split:** split by UE using `GroupShuffleSplit(test_size=0.2)`. This prevents data leakage — the same UE's positions do not appear in both train and test.

**Metrics to compute and report:**

| Metric | Formula |
|---|---|
| Mean Error | `mean(errors)` |
| Median Error | `median(errors)` |
| CEP50 | `percentile(errors, 50)` |
| CEP90 | `percentile(errors, 90)` |
| RMSE | `sqrt(mean(errors²))` |

**Error computation:** Euclidean distance in Cartesian coordinates (meters):

```python
error = np.sqrt((est_x - true_x)**2 + (est_y - true_y)**2)
```

**Hyperparameter tuning:** evaluate K ∈ {1, 3, 5, 7, 9, 15}. Plot mean error vs K. Select K that minimizes mean error on the test set.

**Spatial analysis:** plot localization error as a function of:
- Geographic location (heatmap of error over the study area)
- Number of cells measured per query (error vs. N_cells)
- Distance to nearest BTS

### M4 — Visualization

Four required outputs:

| Output | Description | Format |
|---|---|---|
| Coverage quality map | Mean serving RSRP per grid zone | scatter plot (sim_x/sim_y axes) |
| Localization error map | Error magnitude per test position | scatter plot, color = error |
| CDF of localization error | Cumulative distribution of error | line chart |
| Error vs K | Mean error for each value of K | line chart |

Optional for demo: interactive HTML map using `folium` overlaying estimated positions and BTS locations.

---

## 6. Deliverables

| Deliverable | Description |
|---|---|
| **GitHub repository** | Clean, documented code with README and module structure |
| **EDA notebook** | `notebooks/00_eda.ipynb` with outputs rendered |
| **Evaluation results** | `results/metrics.json` — all metrics for all K values |
| **Visualization outputs** | Static charts (PNG/PDF) + optional interactive HTML map |
| **Written report** | PDF ≤ 25 pages: problem, method, results, discussion, future work |
| **Presentation slides** | ≤ 25 slides for committee presentation |
| **Demo or recording** | Live demo or video of localization results on a map |

---

## 7. Limitations to Acknowledge

### Sim-to-real gap

The fingerprint database is built from **ray tracing simulation**, not from real-world measurements. Simulated RSRP values will differ from real measurements due to:
- Incomplete 3D building geometry (missing small structures, vegetation)
- Approximate material properties (permittivity, conductivity)
- Missing human body and vehicle blockage effects
- Antenna radiation pattern inaccuracies

In practice, this gap increases localization error. The results in this project represent **optimistic upper-bound performance** under ideal simulation conditions.

### No serving/neighbor distinction

Real MRs only report cells that exceed event trigger thresholds (e.g., Event A3), and are limited to `maxCellReport` entries. In this dataset, all measured cells are reported without filtering — query vectors are richer than real-world MRs. The missing neighbor problem (a major challenge in production systems) is not fully reproduced in this dataset.

### No temporal continuity

Each measurement event is localized independently. In a real system, consecutive MRs from the same moving UE could be combined using trajectory smoothing (e.g., Kalman filter) to improve accuracy.

---

## 8. Potential Extensions

These are not required for the base deliverables but represent natural next steps:

**Better missing cell handling.** Compute Euclidean distance only over dimensions present in the query (ignore fill dimensions). Weight candidate fingerprints by the number of matching cell dimensions.

**Learned embedding.** Replace the raw 26-dimensional RSRP vector with a compact embedding learned by an Autoencoder or Siamese/Triplet network. A Triplet loss using geographic distance as supervision produces a metric space where Euclidean distance reflects physical proximity — improving KNN accuracy directly.

**Direct regression.** Train a small MLP to map RSRP vectors directly to `(sim_x, sim_y)` coordinates. Straightforward given the availability of ground truth positions in this dataset.

**Sim-to-real calibration.** Use a small number of real drive-test measurements to learn an offset correction between simulated and measured RSRP, following the approach of Geo2SigMap (Li et al., 2024).