# 0_data_profile.py
"""
Stage 0 — Dataset description (console-only).

This module loads the Parkinson's dataset (CSV) and prints a clean, human-readable
description to stdout:
- dataset overview (shape, missing values)
- column roles (ID / target / features)
- feature typing: quantitative vs categorical (nominal)

Design goals:
- Console output only (no files, no Markdown rendering)
- Reusable `run()` entrypoint for a future orchestrator
- Professional logging + argument parsing
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetTyping:
    """Typed view of dataset columns."""
    id_column: str
    target_column: str
    feature_columns: List[str]
    quantitative_features: List[str]
    categorical_features: List[str]  # inferred among features
    nominal_variables: List[str]  # categorical + id + target


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _infer_feature_types(
        df: pd.DataFrame,
        *,
        id_column: str,
        target_column: str,
        categorical_unique_cap: int = 20,
        categorical_unique_ratio: float = 0.05,
) -> Tuple[List[str], List[str]]:
    """
    Infer feature types from a DataFrame.

    Heuristics:
    - object/string => categorical
    - bool => categorical
    - integer with low cardinality => categorical
    - remaining numeric => quantitative
    """
    feature_cols = [c for c in df.columns if c not in {id_column, target_column}]
    n = len(df)

    categorical: List[str] = []
    quantitative: List[str] = []

    for col in feature_cols:
        s = df[col]
        dtype = s.dtype

        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            categorical.append(col)
            continue

        if pd.api.types.is_bool_dtype(dtype):
            categorical.append(col)
            continue

        if pd.api.types.is_integer_dtype(dtype):
            nunique = int(s.nunique(dropna=True))
            threshold = max(categorical_unique_cap, int(categorical_unique_ratio * n))
            (categorical if nunique <= threshold else quantitative).append(col)
            continue

        if pd.api.types.is_numeric_dtype(dtype):
            quantitative.append(col)
            continue

        # Conservative fallback:
        categorical.append(col)

    return sorted(quantitative), sorted(categorical)


def _format_list_block(items: List[str], *, indent: int = 4, columns: int = 2) -> str:
    """
    Format a list of strings into a neat console block.

    Example:
        - foo                 - bar
        - baz                 - qux
    """
    if not items:
        return " " * indent + "(none)"

    max_len = max(len(x) for x in items)
    col_width = max_len + 6  # "- " + padding

    lines: List[str] = []
    row: List[str] = []

    for i, item in enumerate(items, start=1):
        cell = f"- {item}".ljust(col_width)
        row.append(cell)
        if i % columns == 0:
            lines.append((" " * indent) + "".join(row).rstrip())
            row = []

    if row:
        lines.append((" " * indent) + "".join(row).rstrip())

    return "\n".join(lines)


def _print_report(
        df: pd.DataFrame,
        typing: DatasetTyping,
        *,
        dataset_name: str,
) -> None:
    n_rows, n_cols = df.shape
    missing_total = int(df.isna().sum().sum())
    missing_by_col = df.isna().sum()

    # Console-friendly header
    print("=" * 74)
    print(f"{dataset_name}")
    print("=" * 74)
    print(f"Rows: {n_rows:,}   Columns: {n_cols:,}")
    print(f"Missing values (total): {missing_total}")
    if missing_total:
        top_missing = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
        print("Missing values (by column):")
        for col, cnt in top_missing.items():
            print(f"  - {col}: {int(cnt)}")
    print()

    print("Column roles")
    print("-" * 74)
    print(f"ID column     : {typing.id_column!r}  (categorical, nominal identifier)")
    print(f"Target column : {typing.target_column!r}  (categorical, nominal label)")
    print(f"Features      : {len(typing.feature_columns)} columns")
    print()

    print("Feature typing")
    print("-" * 74)
    print(f"Quantitative (real-valued): {len(typing.quantitative_features)}")
    print(_format_list_block(typing.quantitative_features, columns=2))
    print()
    print(f"Categorical (nominal):     {len(typing.categorical_features)}")
    print(_format_list_block(typing.categorical_features, columns=2))
    print()

    print("Sanity checks")
    print("-" * 74)
    print("Dtypes (all columns):")
    dtypes = df.dtypes.astype(str)
    for col in df.columns:
        print(f"  - {col:<20} {dtypes[col]}")
    print()

    print("Cardinality (unique values, incl. ID/target):")
    nunique = df.nunique(dropna=True)
    for col in df.columns:
        print(f"  - {col:<20} {int(nunique[col])}")
    print()

    print("Notes")
    print("-" * 74)
    print(
        "• The dataset contains multiple recordings per subject (encoded in 'name').\n"
        "  When you evaluate models, prefer group-based splits by subject to avoid leakage.\n"
        "• No ordinal variables are specified in the dataset description; categorical variables\n"
        "  are treated as nominal by default."
    )
    print("=" * 74)


def run(
        input_csv: str | Path,
        *,
        dataset_name: str = "Oxford Parkinson's Disease Detection Dataset",
        id_column: str = "name",
        target_column: str = "status",
) -> DatasetTyping:
    """
    Orchestrator-friendly entrypoint.

    Loads the dataset, infers feature types, prints the report to stdout,
    and returns a structured typing object for downstream stages.
    """
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    if id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' not found. Available: {list(df.columns)}")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Available: {list(df.columns)}")

    feature_cols = [c for c in df.columns if c not in {id_column, target_column}]
    quantitative, categorical = _infer_feature_types(df, id_column=id_column, target_column=target_column)

    nominal = sorted(set(categorical + [id_column, target_column]))

    typing = DatasetTyping(
        id_column=id_column,
        target_column=target_column,
        feature_columns=feature_cols,
        quantitative_features=quantitative,
        categorical_features=categorical,
        nominal_variables=nominal,
    )

    _print_report(df, typing, dataset_name=dataset_name)
    return typing


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 0: Describe dataset columns (console report only)."
    )
    parser.add_argument("--input", required=True, help="Path to parkinsons.data CSV.")
    parser.add_argument("--id-column", default="name", help="ID column name (default: name).")
    parser.add_argument("--target-column", default="status", help="Target column name (default: status).")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)

    run(
        args.input,
        id_column=args.id_column,
        target_column=args.target_column,
    )


if __name__ == "__main__":
    main()
