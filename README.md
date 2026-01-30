# Parkinson's Voice Dataset — EDA & Classification Pipeline

This repository is a small, reproducible pipeline for exploring and modeling the **Oxford Parkinson's Disease Detection Dataset** (UCI ML Repository, dataset id=174, DOI: `10.24432/C59C74`).

## Dataset (what it is)

The dataset contains biomedical voice measurements extracted from speech recordings of **31 subjects**:
- **23** subjects with Parkinson’s disease (PD)
- **8** healthy controls

Each row corresponds to a **single voice recording**. Most subjects have multiple recordings.  
The target is the `status` column:
- `0` — healthy
- `1` — Parkinson’s disease

The `name` column encodes the subject identifier and recording number.

## Project goal (what we are building)

The overall goal is to build a clean pipeline that:
1. **profiles** the dataset and validates assumptions,
2. performs structured **EDA** and feature checks,
3. trains baseline **classification** models for PD detection,
4. evaluates them correctly (i.e., avoiding **subject leakage** by using group-based splits).

Each stage is implemented as a standalone Python module with a `run()` function, so later we can execute all stages from one orchestrator.

---

## Stage 0 — Dataset description (current)

Stage 0 prints a console report that documents:
- dataset shape and missing values,
- column roles (ID / target / features),
- feature typing (quantitative vs categorical).

Run:

```bash
python 0_data_profile.py --input dataset/parkinsons.data
