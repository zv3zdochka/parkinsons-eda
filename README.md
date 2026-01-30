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
1. **Profiles** the dataset and validates assumptions,
2. performs structured **EDA** and feature checks,
3. trains baseline **classification** models for PD detection,
4. evaluates models correctly (group-aware splitting, robust metrics, reproducibility).

---

## Stage 0 — Dataset description (current)

Stage 0 prints a console report documenting:
- dataset shape and missing values,
- column roles (ID / target / features),
- feature typing (quantitative vs categorical).

### Run

```bash
python 0_data_profile.py --input dataset/parkinsons.data
````

### Observed results (from `dataset/parkinsons.data`)

* **Rows:** 195
* **Columns:** 24
* **Missing values:** 0
* **ID column:** `name` (nominal identifier)
* **Target column:** `status` (binary nominal label)
* **Feature columns:** 22
* **Inferred feature typing:**

  * Quantitative (real-valued): **22**
  * Categorical (nominal among features): **0**

### Notes from Stage 0

* Many features have very high cardinality (often unique per row), which is expected for continuous voice measurements.
* `MDVP:Jitter(Abs)` shows low cardinality (19 unique values), likely due to rounding/quantization; this is not an error but worth remembering during EDA.
* The dataset includes multiple recordings per subject (encoded in `name`). For reliable evaluation, later stages should extract a subject ID and use **group-aware splits** (e.g., `GroupKFold`, `GroupShuffleSplit`).

---

## How to run stages

Activate your virtual environment and run any stage script directly:

```bash
python 0_data_profile.py --input dataset/parkinsons.data
```

Later we will add a single orchestrator to run all stages sequentially.


