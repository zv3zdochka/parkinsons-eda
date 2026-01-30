# 0_data_profile.py
"""
Stage 0 — Dataset description (console-only).

What this script does
---------------------
Loads the Parkinson's dataset (CSV) and prints a clean, human-readable report to stdout:
  - Dataset overview (shape, missing values)
  - Column roles (ID / target / features)
  - Feature typing: quantitative vs categorical (nominal)

Why this exists
---------------
We want each project stage to be:
  1) runnable as a standalone script (CLI),
  2) importable and callable from a future orchestrator via a stable `run()` entrypoint,
  3) deterministic and easy to read in console logs.

Notes on this dataset
---------------------
The Parkinson's dataset includes multiple voice recordings per subject (encoded in `name`).
A random row-wise train/test split can leak subject-specific characteristics. Later stages
should prefer group-based splitting by subject (parsed from `name`).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetTyping:
    """
    Typed view of dataset columns.

    Attributes
    ----------
    id_column:
        Identifier column (not a modeling feature).
    target_column:
        Ground-truth label column (not a modeling feature).
    feature_columns:
        All columns used as model inputs (excluding ID and target).
    quantitative_features:
        Feature columns interpreted as numeric (continuous/real-valued).
    categorical_features:
        Feature columns interpreted as categorical.
        (For this dataset we typically expect none among features.)
    nominal_variables:
        All nominal/categorical variables including ID and target.
        Useful for documentation and sanity-checking later stages.
    """
    id_column: str
    target_column: str
    feature_columns: List[str]
    quantitative_features: List[str]
    categorical_features: List[str]
    nominal_variables: List[str]


def _setup_logging(verbosity: int) -> None:
    """
    Configure logging.

    We keep the console report as `print()` output, while logging is reserved for
    operational messages (e.g., file not found, debug info).
    """
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

    Heuristics (simple and intentionally conservative):
      - object/string => categorical
      - bool          => categorical
      - integer with low cardinality => categorical
      - remaining numeric => quantitative

    Parameters
    ----------
    df:
        Input dataframe.
    id_column, target_column:
        Columns that are excluded from feature typing.
    categorical_unique_cap:
        Absolute upper bound for "small-cardinality" integers (e.g., 0/1 flags).
    categorical_unique_ratio:
        Relative upper bound for "small-cardinality" integers (e.g., <= 5% of rows).

    Returns
    -------
    quantitative, categorical:
        Sorted lists of feature column names by inferred type.
    """
    feature_cols = [c for c in df.columns if c not in {id_column, target_column}]
    n_rows = len(df)

    categorical: List[str] = []
    quantitative: List[str] = []

    # Cardinality threshold for treating an integer column as categorical.
    # We take the larger of an absolute cap and a fraction of dataset size.
    int_cat_threshold = max(categorical_unique_cap, int(categorical_unique_ratio * n_rows))

    for col in feature_cols:
        s = df[col]
        dtype = s.dtype

        # Strings / objects are categorical by definition.
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            categorical.append(col)
            continue

        # Booleans are categorical (binary flags).
        if pd.api.types.is_bool_dtype(dtype):
            categorical.append(col)
            continue

        # Integer columns are ambiguous: could be counts or encoded categories.
        # If cardinality is small, treat as categorical; otherwise quantitative.
        if pd.api.types.is_integer_dtype(dtype):
            nunique = int(s.nunique(dropna=True))
            if nunique <= int_cat_threshold:
                categorical.append(col)
            else:
                quantitative.append(col)
            continue

        # Floats and other numeric types default to quantitative.
        if pd.api.types.is_numeric_dtype(dtype):
            quantitative.append(col)
            continue

        # Conservative fallback: if unsure, treat as categorical (safer than numeric).
        categorical.append(col)

    return sorted(quantitative), sorted(categorical)


def _format_list_block(items: List[str], *, indent: int = 4, columns: int = 2) -> str:
    """
    Render a list of strings as a compact, aligned console block.

    Example output (columns=2):
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
        row.append(f"- {item}".ljust(col_width))
        if i % columns == 0:
            lines.append((" " * indent) + "".join(row).rstrip())
            row = []

    if row:
        lines.append((" " * indent) + "".join(row).rstrip())

    return "\n".join(lines)


def _print_report(df: pd.DataFrame, typing: DatasetTyping, *, dataset_name: str) -> None:
    """
    Print the full console report.

    We keep formatting stable so later you can compare logs across runs/branches.
    """
    n_rows, n_cols = df.shape

    # Missing values summary
    missing_total = int(df.isna().sum().sum())
    missing_by_col = df.isna().sum()

    print("=" * 74)
    print(dataset_name)
    print("=" * 74)
    print(f"Rows: {n_rows:,}   Columns: {n_cols:,}")
    print(f"Missing values (total): {missing_total}")
    if missing_total:
        print("Missing values (by column):")
        top_missing = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
        for col, cnt in top_missing.items():
            print(f"  - {col}: {int(cnt)}")
    print()

    # Column roles
    print("Column roles")
    print("-" * 74)
    print(f"ID column     : {typing.id_column!r}  (categorical, nominal identifier)")
    print(f"Target column : {typing.target_column!r}  (categorical, nominal label)")
    print(f"Features      : {len(typing.feature_columns)} columns")
    print()

    # Feature typing
    print("Feature typing")
    print("-" * 74)
    print(f"Quantitative (real-valued): {len(typing.quantitative_features)}")
    print(_format_list_block(typing.quantitative_features, columns=2))
    print()
    print(f"Categorical (nominal):     {len(typing.categorical_features)}")
    print(_format_list_block(typing.categorical_features, columns=2))
    print()

    # Sanity checks: dtypes and cardinality
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

    # Practical notes that matter for evaluation correctness
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

    Loads the dataset, infers feature types, prints the report to stdout, and returns a
    `DatasetTyping` object so downstream stages can reuse the typing information.

    Parameters
    ----------
    input_csv:
        Path to the CSV file (e.g., dataset/parkinsons.data).
    dataset_name:
        Friendly dataset name used in the report header.
    id_column:
        Identifier column (not used as a model feature).
    target_column:
        Label column (not used as a model feature).

    Returns
    -------
    DatasetTyping:
        Structured typing output suitable for reuse in later stages.
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

    # Nominal variables = inferred categorical feature columns + explicitly nominal ID/target.
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
    """CLI contract for stage execution."""
    parser = argparse.ArgumentParser(
        description="Stage 0: Describe dataset columns (console report only)."
    )
    parser.add_argument("--input", required=True, help="Path to parkinsons.data CSV.")
    parser.add_argument("--id-column", default="name", help="ID column name (default: name).")
    parser.add_argument("--target-column", default="status", help="Target column name (default: status).")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv).")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = _parse_args()
    _setup_logging(args.verbose)

    run(
        args.input,
        id_column=args.id_column,
        target_column=args.target_column,
    )


if __name__ == "__main__":
    main()
