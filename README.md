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


