# 1_missing_values.py
"""
Stage 1 — Missing values audit & handling (console-only).

Task
----
1) Check whether the dataset has missing values.
2) Compute missingness percentage relative to the number of rows.
3) Explain plausible reasons for missingness (if any).
4) If missingness exists:
   - choose an imputation/handling strategy per column,
   - justify the choice,
   - apply the treatment and report the final status.

Design goals
------------
- Console output only (no markdown generation).
- Orchestrator-friendly `run()` entrypoint.
- Deterministic, explicit, and readable.

Notes
-----
Even if missingness is zero (as in the canonical UCI Parkinson's CSV), we still implement
the full machinery so the pipeline remains robust to other dataset versions.
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
class MissingnessSummary:
    """Summary of missing values statistics."""
    n_rows: int
    n_cols: int
    total_missing: int
    total_missing_pct_of_rows: float
    missing_by_column: Dict[str, int]
    missing_pct_by_column_of_rows: Dict[str, float]


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _compute_missingness(df: pd.DataFrame) -> MissingnessSummary:
    n_rows, n_cols = df.shape
    missing_by_col = df.isna().sum().astype(int).to_dict()
    total_missing = int(sum(missing_by_col.values()))
    total_missing_pct_of_rows = (total_missing / n_rows * 100.0) if n_rows else 0.0

    missing_pct_by_col_of_rows = {
        col: (cnt / n_rows * 100.0) if n_rows else 0.0
        for col, cnt in missing_by_col.items()
    }

    return MissingnessSummary(
        n_rows=n_rows,
        n_cols=n_cols,
        total_missing=total_missing,
        total_missing_pct_of_rows=total_missing_pct_of_rows,
        missing_by_column=missing_by_col,
        missing_pct_by_column_of_rows=missing_pct_by_col_of_rows,
    )


def _infer_imputation_plan(
        df: pd.DataFrame,
        *,
        id_column: str,
        target_column: str,
) -> Dict[str, str]:
    """
    Create a per-column missing value handling plan.

    Strategies (simple and defensible):
    - ID column:
        -> "drop_rows" (cannot reliably impute identity)
    - Target column:
        -> "drop_rows" (cannot impute labels without leakage)
    - Numeric columns:
        -> "median" (robust to outliers; common for biomedical measures)
    - Categorical columns:
        -> "most_frequent" (mode)
    """
    plan: Dict[str, str] = {}

    for col in df.columns:
        if col == id_column:
            plan[col] = "drop_rows"
            continue
        if col == target_column:
            plan[col] = "drop_rows"
            continue

        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            plan[col] = "median"
        else:
            plan[col] = "most_frequent"

    return plan


def _apply_missing_value_treatment(
        df: pd.DataFrame,
        plan: Dict[str, str],
) -> Tuple[pd.DataFrame, Dict[str, str], int]:
    """
    Apply the treatment plan.

    Returns
    -------
    cleaned_df:
        A copy of the input with missing values handled.
    applied_actions:
        Human-readable description of what happened per column.
    rows_dropped:
        Number of rows removed (if any).
    """
    cleaned = df.copy(deep=True)
    applied_actions: Dict[str, str] = {}
    rows_before = len(cleaned)

    # 1) Drop rows if needed (ID/target typically)
    drop_cols = [c for c, strat in plan.items() if strat == "drop_rows"]
    if drop_cols:
        # Drop rows where any of those critical columns are missing.
        cleaned = cleaned.dropna(subset=drop_cols)
        applied_actions.update({c: "Dropped rows where column is missing" for c in drop_cols})

    rows_after_drop = len(cleaned)
    rows_dropped = rows_before - rows_after_drop

    # 2) Impute remaining columns
    for col, strat in plan.items():
        if strat == "drop_rows":
            continue

        if cleaned[col].isna().sum() == 0:
            applied_actions[col] = "No missing values"
            continue

        if strat == "median":
            med = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(med)
            applied_actions[col] = f"Filled missing values with median = {med!r}"
        elif strat == "most_frequent":
            # mode() can return multiple values; take the first deterministic one
            mode_series = cleaned[col].mode(dropna=True)
            fill_value = mode_series.iloc[0] if not mode_series.empty else ""
            cleaned[col] = cleaned[col].fillna(fill_value)
            applied_actions[col] = f"Filled missing values with most frequent value = {fill_value!r}"
        else:
            raise ValueError(f"Unknown strategy '{strat}' for column '{col}'")

    return cleaned, applied_actions, rows_dropped


def _print_report(
        *,
        dataset_name: str,
        before: MissingnessSummary,
        after: MissingnessSummary,
        plan: Dict[str, str],
        applied_actions: Dict[str, str],
        rows_dropped: int,
) -> None:
    print("=" * 74)
    print(f"{dataset_name} — Stage 1: Missing values")
    print("=" * 74)

    print(f"Rows: {before.n_rows:,}   Columns: {before.n_cols:,}")
    print()

    print("1) Missingness overview (before)")
    print("-" * 74)
    print(f"Total missing cells: {before.total_missing:,}")
    print(f"Missing cells as % of rows: {before.total_missing_pct_of_rows:.6f}%")
    # Show only columns with missingness
    missing_cols = [(c, before.missing_by_column[c], before.missing_pct_by_column_of_rows[c])
                    for c in before.missing_by_column if before.missing_by_column[c] > 0]
    if not missing_cols:
        print("Columns with missing values: none")
    else:
        print("Columns with missing values:")
        for col, cnt, pct in sorted(missing_cols, key=lambda x: (-x[1], x[0])):
            print(f"  - {col:<20} {cnt:>6} missing  ({pct:.4f}% of rows)")
    print()

    print("2) Why missing values may appear (general reasons)")
    print("-" * 74)
    print(
        "Typical sources of missingness in biomedical measurement datasets:\n"
        "  • Sensor/recording artifacts (clipped signal, noise, failed extraction)\n"
        "  • Preprocessing failures for specific recordings (feature not computable)\n"
        "  • Data integration issues (manual entry, merges, inconsistent identifiers)\n"
        "  • File format/version differences across dataset releases\n"
        "\n"
        "For the canonical UCI Parkinson's CSV, missingness is usually zero; however the pipeline\n"
        "keeps the handling logic to remain robust across variants."
    )
    print()

    print("3) Chosen handling strategy (per column type)")
    print("-" * 74)
    print("Strategy rules:")
    print("  • ID / target: drop rows if missing (cannot be imputed safely)")
    print("  • numeric features: median imputation (robust to outliers)")
    print("  • categorical features: most frequent value (mode)")
    print()

    # Print only strategies for columns that actually had missing values OR are critical (drop rules)
    print("Applied plan (relevant columns):")
    relevant = []
    for col, strat in plan.items():
        if strat == "drop_rows":
            relevant.append((col, strat))
        elif before.missing_by_column.get(col, 0) > 0:
            relevant.append((col, strat))

    if not relevant:
        print("  (no missing values detected; no imputation required)")
    else:
        for col, strat in sorted(relevant, key=lambda x: (x[1], x[0])):
            print(f"  - {col:<20} -> {strat}")
    print()

    print("4) Treatment summary")
    print("-" * 74)
    if rows_dropped:
        print(f"Rows dropped due to missing ID/target: {rows_dropped}")
    else:
        print("Rows dropped due to missing ID/target: 0")

    # If there were missing values, show what happened per affected column
    affected_cols = [c for c in applied_actions if "Filled missing values" in applied_actions[c]]
    if not affected_cols:
        print("Imputation actions: none (no missing values to fill)")
    else:
        print("Imputation actions:")
        for col in sorted(affected_cols):
            print(f"  - {col}: {applied_actions[col]}")
    print()

    print("5) Missingness overview (after)")
    print("-" * 74)
    print(f"Total missing cells: {after.total_missing:,}")
    print(f"Missing cells as % of rows: {after.total_missing_pct_of_rows:.6f}%")
    print()

    if after.total_missing != 0:
        print("WARNING: Missing values remain after treatment (investigate).")
    else:
        print("OK: No missing values remain after treatment.")
    print("=" * 74)


def run(
        input_csv: str | Path,
        *,
        dataset_name: str = "Oxford Parkinson's Disease Detection Dataset",
        id_column: str = "name",
        target_column: str = "status",
        output_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Orchestrator-friendly entrypoint.

    Loads the dataset, audits missing values, applies a defensible handling strategy,
    prints a full report, and optionally saves a cleaned CSV.

    Parameters
    ----------
    input_csv:
        Path to the raw CSV.
    output_csv:
        Optional path to save the cleaned dataset.
        If None, no file is written.
    id_column, target_column:
        Critical columns. If missing values exist there, affected rows are dropped.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset (missing values handled).
    """
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    if id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' not found. Available: {list(df.columns)}")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Available: {list(df.columns)}")

    before = _compute_missingness(df)
    plan = _infer_imputation_plan(df, id_column=id_column, target_column=target_column)
    cleaned_df, applied_actions, rows_dropped = _apply_missing_value_treatment(df, plan=plan)
    after = _compute_missingness(cleaned_df)

    _print_report(
        dataset_name=dataset_name,
        before=before,
        after=after,
        plan=plan,
        applied_actions=applied_actions,
        rows_dropped=rows_dropped,
    )

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(output_csv, index=False)
        LOGGER.info("Saved cleaned dataset to: %s", output_csv)

    return cleaned_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: Missing values audit & handling.")
    parser.add_argument("--input", required=True, help="Path to parkinsons.data CSV.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write a cleaned CSV (e.g., dataset/parkinsons_clean.csv).",
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
        output_csv=args.output,
        id_column=args.id_column,
        target_column=args.target_column,
    )


if __name__ == "__main__":
    main()
