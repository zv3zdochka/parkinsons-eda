# Parkinson's Voice Dataset — EDA & Classification Pipeline

This repository implements a small, reproducible pipeline for exploring and modeling the **Oxford Parkinson's Disease Detection Dataset** (UCI ML Repository, dataset id=174, DOI: `10.24432/C59C74`).

The codebase is organized as a sequence of **numbered stages** (`0_*.py`, `1_*.py`, ...).  
Each stage is runnable via CLI and exposes a stable `run()` entrypoint so we can later execute all stages from a single orchestrator.

---

## Dataset (what it is)

The dataset contains biomedical voice measurements extracted from speech recordings of **31 subjects**:
- **23** subjects with Parkinson’s disease (PD)
- **8** healthy controls

Each row corresponds to a **single voice recording**. Most subjects have multiple recordings.  
The classification target is the `status` column:
- `0` — healthy
- `1` — Parkinson’s disease

The `name` column encodes the subject identifier and recording number.

**Important:** Because there are multiple recordings per subject, **row-wise random splits can cause subject leakage**. Proper evaluation should prefer **group-based splits by subject**.

---

## Project goal (what we are building)

The overall goal is to build a clean pipeline that:
1. Profiles the dataset and validates assumptions,
2. performs structured EDA and feature checks,
3. trains baseline classification models for PD detection,
4. evaluates models correctly (group-aware splitting, robust metrics, reproducibility).

---

## Stage 0 — Dataset description (answers TЗ #0)

**TЗ question:**  
0. Provide a dataset description. Which features are quantitative, categorical, nominal?

### Run

```bash
python 0_data_profile.py --input dataset/parkinsons.data
````

### Observed results (from `dataset/parkinsons.data`)

* **Rows:** 195
* **Columns:** 24
* **Missing values:** 0
* **ID column:** `name` (categorical, **nominal identifier**)
* **Target column:** `status` (categorical, **nominal label**)
* **Feature columns:** 22

**Feature types:**

* **Quantitative (real-valued): 22**
  All modeling features are numeric biomedical voice measurements (floats).
* **Categorical among features: 0**
  No categorical predictors are present in this dataset version.
* **Nominal variables:**
  `name` (ID) and `status` (binary target) are nominal.

### Notes from Stage 0

* Many features have very high cardinality (often unique per row), which is expected for continuous voice measurements.
* `MDVP:Jitter(Abs)` has low cardinality (19 unique values), likely due to rounding/quantization.
* Multiple recordings per subject are encoded in `name`; later stages should extract a subject ID and use **group-aware splits**.

---

## Stage 1 — Missing values audit & handling (answers TЗ #1)

**TЗ question:**

1. Are there missing values? What percentage do they represent relative to the number of rows? Why might they appear?
   If missing values exist, choose a handling method per column, justify it, and process missing values.

### Run

```bash
python 1_missing_values.py --input dataset/parkinsons.data --output dataset/parkinsons_clean.csv
```

### Observed results (from `dataset/parkinsons.data`)

* **Total missing cells:** 0
* **Missingness as % of rows:** **0.000000%**
* **Columns with missing values:** none
* A cleaned copy was written to: `dataset/parkinsons_clean.csv`
  (no changes were necessary, but this file is used as a stable input for later stages).

### Why missing values may appear (general reasons)

Even though this dataset version contains no missing values, typical sources of missingness in biomedical measurement data include:

* recording/sensor artifacts (noise, clipping, corrupted audio),
* preprocessing/feature-extraction failures (feature not computable for a recording),
* data integration issues (manual entry, merges, inconsistent identifiers),
* dataset version differences across releases.

### Handling strategy (defined for robustness)

If missingness were present, the pipeline would apply the following defensible rules:

* **ID (`name`) and target (`status`):** drop rows if missing
  (identity and labels cannot be imputed safely without introducing errors/leakage)
* **Numeric features:** median imputation
  (robust to outliers; appropriate for biomedical signals)
* **Categorical features:** most frequent value (mode)
  (standard baseline for nominal features)

**In this run:** no imputation and no row drops were required.

---

## How to run stages

```bash
python 0_data_profile.py --input dataset/parkinsons.data
python 1_missing_values.py --input dataset/parkinsons.data --output dataset/parkinsons_clean.csv
```

Later we will add a single orchestrator to run all stages sequentially.


## Stage 2 — Type checks + descriptive statistics (answers TЗ #2 and #3)

**TЗ questions:**
2. Do the dataset dtypes match what we need (e.g., no numeric variables stored as strings)? Fix issues if needed.  
3. Compute descriptive statistics for all variables.

### Run

```bash
python 2_types_and_stats.py --input dataset/parkinsons_clean.csv --output dataset/parkinsons_typed.csv
````

### Observed results (from `dataset/parkinsons_clean.csv`)

**Type validation (Task #2)**

* Rows: **195 → 195**, Columns: **24**
* `name` (ID): `str` → **pandas `string`** (normalized for stable text handling)
* `status` (target): **`int64`** (validated binary labels and kept as integer)
* Non-numeric feature columns detected: **0**
* Rows dropped due to missing ID/target: **0**
* Missing values after coercion: **none** (no imputation needed)

A schema-normalized copy was saved to: `dataset/parkinsons_typed.csv`.

**Descriptive statistics (Task #3)**

* Target distribution:

  * `status=0`: **48** rows (**24.62%**)
  * `status=1`: **147** rows (**75.38%**)

* Numeric summary statistics were computed for all 22 real-valued voice features
  (count/mean/std/min/25%/50%/75%/max).

* Nominal summaries were computed for `name` (all unique) and `status` (binary).

### Notes

* The dataset is **class-imbalanced** (≈75% PD). Later modeling stages should consider metrics
  beyond raw accuracy (e.g., ROC-AUC, PR-AUC, balanced accuracy) and proper validation.
* Some variables exhibit long-tailed ranges (e.g., `NHR`, `MDVP:Fhi(Hz)`), which is typical for
  biomedical measurements and may motivate robust scaling or outlier-aware analysis later.

---

## Stage 3 — Feature inspection, outliers/errors, correlation (answers TЗ #4–#6)

**TЗ questions:**
4. Inspect each feature. What values does it take?
   - numeric: value range
   - categorical: unique values
5. Are there outliers or errors (e.g., negative price)? How do you define them? Handle outliers.
6. Build the correlation matrix for numeric variables.

### Run

```bash
python 3_feature_audit_outliers_corr.py --input dataset/parkinsons_typed.csv --output dataset/parkinsons_stage3.csv --corr-output reports/corr_stage3.csv
````

### Observed results (from `dataset/parkinsons_typed.csv`)

**Task #4 — Feature inspection**

* Numeric features: **22**
* Nominal/categorical columns: **2**

  * `name`: **195 unique** (identifier, one per recording)
  * `status`: **2 unique** values (0, 1)

Numeric feature ranges (min → max):

* `MDVP:Fo(Hz)`: 88.333 → 260.105
* `MDVP:Fhi(Hz)`: 102.145 → 592.030
* `MDVP:Flo(Hz)`: 65.476 → 239.170
* `MDVP:Jitter(%)`: 0.001680 → 0.033160
* `MDVP:Jitter(Abs)`: 0.000007 → 0.000260
* `MDVP:RAP`: 0.000680 → 0.021440
* `MDVP:PPQ`: 0.000920 → 0.019580
* `Jitter:DDP`: 0.002040 → 0.064330
* `MDVP:Shimmer`: 0.009540 → 0.119080
* `MDVP:Shimmer(dB)`: 0.085 → 1.302
* `Shimmer:APQ3`: 0.004550 → 0.056470
* `Shimmer:APQ5`: 0.005700 → 0.079400
* `MDVP:APQ`: 0.007190 → 0.137780
* `Shimmer:DDA`: 0.013640 → 0.169420
* `NHR`: 0.000650 → 0.314820
* `HNR`: 8.441 → 33.047
* `RPDE`: 0.256570 → 0.685151
* `DFA`: 0.574282 → 0.825288
* `spread1`: -7.964984 → -2.434031
* `spread2`: 0.006274 → 0.450493
* `D2`: 1.423287 → 3.671155
* `PPE`: 0.044539 → 0.527367

**Task #5 — Outliers / errors**
Definitions used:

* **Errors (domain violations):** values that violate conservative domain rules (e.g., negative frequencies, `RPDE`/`DFA` outside [0,1]).
  Treatment: set to NaN → median imputation (numeric).
* **Outliers (statistical extremes):** Tukey IQR rule with **k=1.5** (outside `[Q1 - 1.5·IQR, Q3 + 1.5·IQR]`).
  Treatment: **winsorization** (clip to the IQR fences), to preserve the small dataset size.

Observed:

* Domain violations detected: **none**
* IQR outliers detected (before winsorization): **173 cells**
  Most affected columns:

  * `NHR` (19), `MDVP:PPQ` (15), `Jitter:DDP` (14), `MDVP:Jitter(%)` (14), `MDVP:RAP` (14),
  * `Shimmer:APQ5` (13), `MDVP:APQ` (12), `MDVP:Fhi(Hz)` (11), `MDVP:Shimmer(dB)` (10), `MDVP:Flo(Hz)` (9), ...
* Winsorized (clipped) cells: **173**
* Median imputation: **not needed** (no invalid/missing values after checks)

A cleaned dataset after winsorization was saved to: `dataset/parkinsons_stage3.csv`.

**Task #6 — Correlation matrix**

* Pearson correlation matrix was computed for **all numeric features** (target excluded) and saved to:
  `reports/corr_stage3.csv`.

Top 15 correlated feature pairs (by |corr|):

* `MDVP:RAP` vs `Jitter:DDP`: **1.000**
* `Shimmer:APQ3` vs `Shimmer:DDA`: **1.000**
* `MDVP:Shimmer` vs `MDVP:Shimmer(dB)`: **0.992**
* `MDVP:Shimmer` vs `Shimmer:APQ3`: **0.990**
* `MDVP:Shimmer` vs `Shimmer:DDA`: **0.990**
* `MDVP:Shimmer` vs `Shimmer:APQ5`: **0.985**
* `MDVP:Jitter(%)` vs `Jitter:DDP`: **0.979**
* `MDVP:Jitter(%)` vs `MDVP:RAP`: **0.979**
* `MDVP:Shimmer(dB)` vs `Shimmer:APQ5`: **0.978**
* `MDVP:Shimmer(dB)` vs `Shimmer:APQ3`: **0.976**
* `MDVP:Shimmer(dB)` vs `Shimmer:DDA`: **0.976**
* `MDVP:Shimmer` vs `MDVP:APQ`: **0.970**
* `Shimmer:APQ3` vs `Shimmer:APQ5`: **0.969**
* `Shimmer:APQ5` vs `Shimmer:DDA`: **0.969**
* `MDVP:Jitter(%)` vs `MDVP:PPQ`: **0.968**

### Notes

* Several feature groups are **near-duplicates** (correlations ≈ 0.97–1.00), reflecting that many jitter/shimmer measures are derived from closely related formulas.
* Winsorization reduces the influence of extreme values while preserving the dataset size; later modeling may additionally benefit from scaling and/or feature selection to address multicollinearity.
