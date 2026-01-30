# 2_types_and_stats.py
"""
Stage 2 — Schema validation (types) + descriptive statistics (console-only).

Tasks covered (from the spec)
----------------------------
2) Verify that dataset dtypes match what we need (e.g., no numeric variables stored as strings).
   If issues exist, fix them.

3) Compute descriptive statistics for all variables.

What this stage does
--------------------
- Loads a CSV.
- Validates column roles:
    * ID column (`name`) must be string-like
    * target (`status`) must be integer-like with values in {0, 1}
    * all feature columns should be numeric
- If it finds feature columns stored as non-numeric (object/string), it attempts to convert them
  using `pd.to_numeric(errors="coerce")` and reports how many values could not be converted.
- If conversion introduces missing values, it handles them with a defensible baseline:
    * missing ID/target => drop rows
    * missing numeric features => median imputation
    * missing categorical features => mode imputation
- Prints a clean console report:
    * schema / dtype checks (before & after)
    * status distribution
    * descriptive stats for numeric variables
    * descriptive stats for categorical variables (incl. ID/target)

Design goals
------------
- Console output only.
- Orchestrator-friendly `run()` entrypoint.
- Deterministic, readable reporting.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TypeFixReport:
    """Structured summary of dtype checks and fixes."""
    rows_before: int
    rows_after: int
    cols: int

    dtype_before: Dict[str, str]
    dtype_after: Dict[str, str]

    feature_cols: List[str]
    non_numeric_feature_cols_before: List[str]
    coercion_failures_by_col: Dict[str, int]  # values that became NaN after coercion

    rows_dropped_due_to_missing_id_or_target: int
    imputed_numeric_cols: List[str]
    imputed_categorical_cols: List[str]


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _dtype_map(df: pd.DataFrame) -> Dict[str, str]:
    return {c: str(df[c].dtype) for c in df.columns}


def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s.dtype)


def _safe_mode(series: pd.Series):
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else ""


def _normalize_schema(
    df: pd.DataFrame,
    *,
    id_column: str,
    target_column: str,
) -> Tuple[pd.DataFrame, TypeFixReport]:
    """
    Validate and normalize dataset dtypes.

    - Ensures ID is string.
    - Ensures target is int with values in {0,1}.
    - Ensures all features are numeric; attempts coercion if needed.
    - If coercion introduces missing values, applies baseline handling.

    Returns
    -------
    normalized_df, report
    """
    if id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' not found. Available: {list(df.columns)}")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Available: {list(df.columns)}")

    rows_before, n_cols = df.shape
    dtype_before = _dtype_map(df)

    work = df.copy(deep=True)

    # Column roles
    feature_cols = [c for c in work.columns if c not in {id_column, target_column}]

    # 1) ID: enforce string-like
    # Use pandas "string" dtype for clean handling and consistent printing.
    work[id_column] = work[id_column].astype("string")

    # 2) Target: enforce integer-like with values {0,1}
    # If target is not numeric (e.g., strings), coerce.
    if not _is_numeric_series(work[target_column]):
        coerced = pd.to_numeric(work[target_column], errors="coerce")
        work[target_column] = coerced

    # If target came as float (common in CSVs), keep as integer where possible.
    # Missing targets are handled by dropping rows below.
    # Validate values once we have a numeric representation.
    # (We validate later after potential row drops.)
    # 3) Features: ensure numeric
    non_numeric_features = [c for c in feature_cols if not _is_numeric_series(work[c])]
    coercion_failures: Dict[str, int] = {}

    for col in non_numeric_features:
        before_na = int(work[col].isna().sum())
        coerced = pd.to_numeric(work[col], errors="coerce")
        after_na = int(coerced.isna().sum())
        coercion_failures[col] = max(0, after_na - before_na)
        work[col] = coerced

    # 4) Handle missingness introduced by coercion (and any pre-existing)
    rows_pre_drop = len(work)
    work = work.dropna(subset=[id_column, target_column])
    rows_dropped = rows_pre_drop - len(work)

    # Validate target values now that missing targets are removed.
    # We do not silently "fix" wrong labels; raise if unexpected values appear.
    # However, allow 0/1 stored as floats.
    uniq = set(work[target_column].unique().tolist())
    if not uniq.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(
            f"Unexpected target values in '{target_column}': {sorted(uniq)}. "
            "Expected binary labels {0,1}."
        )

    # Cast target to int for cleanliness.
    work[target_column] = work[target_column].astype(int)

    # Impute remaining missing values
    imputed_numeric: List[str] = []
    imputed_categorical: List[str] = []

    for col in work.columns:
        if col in {id_column, target_column}:
            continue
        if work[col].isna().sum() == 0:
            continue

        if _is_numeric_series(work[col]):
            med = work[col].median()
            work[col] = work[col].fillna(med)
            imputed_numeric.append(col)
        else:
            fill = _safe_mode(work[col])
            work[col] = work[col].fillna(fill)
            imputed_categorical.append(col)

    dtype_after = _dtype_map(work)

    report = TypeFixReport(
        rows_before=rows_before,
        rows_after=len(work),
        cols=n_cols,
        dtype_before=dtype_before,
        dtype_after=dtype_after,
        feature_cols=feature_cols,
        non_numeric_feature_cols_before=sorted(non_numeric_features),
        coercion_failures_by_col=coercion_failures,
        rows_dropped_due_to_missing_id_or_target=rows_dropped,
        imputed_numeric_cols=sorted(imputed_numeric),
        imputed_categorical_cols=sorted(imputed_categorical),
    )
    return work, report


def _print_schema_report(report: TypeFixReport, *, id_column: str, target_column: str) -> None:
    print("=" * 74)
    print("Stage 2: Type checks + descriptive statistics")
    print("=" * 74)
    print(f"Rows: {report.rows_before:,} -> {report.rows_after:,}   Columns: {report.cols:,}")
    print()

    print("2) Type validation (before)")
    print("-" * 74)
    print(f"ID column     : {id_column:<20} dtype={report.dtype_before[id_column]}")
    print(f"Target column : {target_column:<20} dtype={report.dtype_before[target_column]}")
    print()

    # Feature dtype issues
    print(f"Non-numeric feature columns (before): {len(report.non_numeric_feature_cols_before)}")
    if report.non_numeric_feature_cols_before:
        for col in report.non_numeric_feature_cols_before:
            print(f"  - {col:<20} dtype={report.dtype_before[col]}")
    else:
        print("  (none)")
    print()

    if report.non_numeric_feature_cols_before:
        print("Coercion results (object/string -> numeric)")
        print("-" * 74)
        for col in report.non_numeric_feature_cols_before:
            fails = report.coercion_failures_by_col.get(col, 0)
            msg = f"{fails} values became NaN during coercion" if fails else "clean conversion (no new NaNs)"
            print(f"  - {col:<20} {msg}")
        print()

    print("Fixes applied")
    print("-" * 74)
    print("ID column:")
    print("  - Enforced pandas 'string' dtype for stable text handling.")
    print("Target column:")
    print("  - Coerced to numeric if needed; validated binary labels; cast to int.")
    if report.rows_dropped_due_to_missing_id_or_target:
        print(f"Rows dropped (missing ID/target): {report.rows_dropped_due_to_missing_id_or_target}")
    else:
        print("Rows dropped (missing ID/target): 0")

    if report.imputed_numeric_cols or report.imputed_categorical_cols:
        print("Missing value handling after coercion:")
        if report.imputed_numeric_cols:
            print(f"  - Numeric: median imputation for {len(report.imputed_numeric_cols)} columns")
        if report.imputed_categorical_cols:
            print(f"  - Categorical: mode imputation for {len(report.imputed_categorical_cols)} columns")
    else:
        print("Missing value handling after coercion: not needed (no missing values).")
    print()

    print("2) Type validation (after)")
    print("-" * 74)
    print(f"ID column     : {id_column:<20} dtype={report.dtype_after[id_column]}")
    print(f"Target column : {target_column:<20} dtype={report.dtype_after[target_column]}")
    print("Features: numeric dtypes expected.")
    print()


def _print_descriptive_statistics(df: pd.DataFrame, *, id_column: str, target_column: str) -> None:
    print("3) Descriptive statistics")
    print("-" * 74)

    # Status distribution is critical for classification context.
    print("Target distribution (`status`):")
    vc = df[target_column].value_counts(dropna=False).sort_index()
    pct = (vc / len(df) * 100.0).round(2)
    for k in vc.index:
        print(f"  - {target_column}={int(k)}: {int(vc.loc[k]):>4} rows ({float(pct.loc[k]):>6.2f}%)")
    print()

    # Numeric stats for all numeric columns (including the target is optional; we exclude it here).
    numeric_cols = [c for c in df.columns if c != target_column and pd.api.types.is_numeric_dtype(df[c].dtype)]
    num_desc = df[numeric_cols].describe().T  # per-variable rows
    # Improve readability: consistent rounding and ordering
    num_desc = num_desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].round(6)

    print("Numeric variables (summary):")
    print(num_desc.to_string())
    print()

    # Categorical stats for non-numeric columns + target as categorical
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    # Ensure target is included in categorical summary (even though it's int)
    if target_column not in cat_cols:
        cat_cols.append(target_column)

    # `describe(include="all")` can be noisy; better: per-categorical describe.
    # Here: count/unique/top/freq for categorical-like columns.
    # Convert target to string for this summary so unique/top are meaningful.
    cat_frame = df[cat_cols].copy()
    if target_column in cat_frame.columns:
        cat_frame[target_column] = cat_frame[target_column].astype("string")

    cat_desc = cat_frame.describe(include=["object", "string"]).T
    print("Categorical / nominal variables (summary):")
    print(cat_desc.to_string())
    print()


def run(
    input_csv: str | Path,
    *,
    id_column: str = "name",
    target_column: str = "status",
    output_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Orchestrator-friendly entrypoint.

    Parameters
    ----------
    input_csv:
        Path to the dataset CSV (recommended: output from Stage 1).
    output_csv:
        Optional path to save a schema-normalized copy.

    Returns
    -------
    pd.DataFrame
        Normalized dataset.
    """
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    normalized, report = _normalize_schema(df, id_column=id_column, target_column=target_column)

    _print_schema_report(report, id_column=id_column, target_column=target_column)
    _print_descriptive_statistics(normalized, id_column=id_column, target_column=target_column)

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        normalized.to_csv(output_csv, index=False)
        LOGGER.info("Saved schema-normalized dataset to: %s", output_csv)

    return normalized


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: Type checks + descriptive statistics.")
    parser.add_argument("--input", required=True, help="Path to dataset CSV.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write a schema-normalized CSV (e.g., dataset/parkinsons_typed.csv).",
    )
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
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
