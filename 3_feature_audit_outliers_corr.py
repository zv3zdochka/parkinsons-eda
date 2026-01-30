# 3_feature_audit_outliers_corr.py
"""
Stage 3 — Feature audit, outliers/errors handling, and correlation matrix (console-first).

Tasks covered (from the spec)
----------------------------
4) Inspect each feature:
   - numeric features: value range (min/max)
   - categorical features: unique values

5) Detect outliers or obvious errors (e.g., negative values where impossible).
   Explain how outliers are defined and handle them.

6) Build a correlation matrix for numeric variables.

Approach
--------
A) Feature audit
   - Numeric: min/max + a few robust stats
   - Categorical: unique count + values (only if small), otherwise preview

B) Errors & outliers
   We separate "errors" from "outliers":
   - Errors: domain violations (e.g., negative frequency, RPDE outside [0,1]).
     These are treated as invalid values -> set to NaN -> impute with median (numeric).
   - Outliers: extreme but plausible values detected via Tukey's IQR rule
     (outside [Q1 - k*IQR, Q3 + k*IQR]).
     We *winsorize* (clip) to the fence, because:
       • dataset is small (195 rows)
       • dropping rows would throw away information
       • clipping is deterministic and preserves row count

C) Correlations
   - Pearson correlation for numeric *features* (target excluded)
   - Print full matrix and also top correlated pairs by absolute correlation.

Outputs
-------
- Console report (primary)
- Optional cleaned CSV output
- Optional correlation matrix CSV output
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutlierHandlingSummary:
    """Structured summary of what was detected and changed."""
    numeric_features: List[str]
    categorical_columns: List[str]

    domain_violations: Dict[str, int]          # invalid values replaced with NaN
    iqr_outliers_before: Dict[str, int]        # values outside IQR fences (before clipping)
    iqr_winsorized_cells: Dict[str, int]       # count of values clipped to fences

    imputed_numeric_columns: List[str]


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series.dtype)


def _safe_mode(series: pd.Series):
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else ""


def _preview_unique(values: Sequence, *, max_items: int = 20) -> str:
    """
    Return a readable representation of unique values.
    - if unique count <= max_items: show all values
    - else: show first 10 + last 3
    """
    vals = list(values)
    n = len(vals)
    if n == 0:
        return "(none)"
    if n <= max_items:
        return ", ".join(repr(v) for v in vals)

    head = vals[:10]
    tail = vals[-3:]
    return f"{', '.join(repr(v) for v in head)}, ... , {', '.join(repr(v) for v in tail)} (total={n})"


def _numeric_range_block(df: pd.DataFrame, cols: List[str]) -> str:
    """Return a compact table of ranges for numeric columns."""
    stats = pd.DataFrame({
        "min": df[cols].min(),
        "max": df[cols].max(),
        "mean": df[cols].mean(),
        "std": df[cols].std(),
    }).round(6)

    # Keep a stable column order.
    stats = stats[["min", "max", "mean", "std"]]
    return stats.to_string()


def _build_domain_rules() -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """
    Domain rules for "obvious errors" (not statistical outliers).

    A value violating these bounds is treated as invalid:
      -> replaced with NaN -> later imputed with median.

    Notes:
    - spread1 is allowed to be negative (as observed in the dataset).
    - Bounds are conservative; they're meant to catch clear nonsense, not to over-police data.
    """
    return {
        # Frequencies must be positive
        "MDVP:Fo(Hz)": (0.0, None),
        "MDVP:Fhi(Hz)": (0.0, None),
        "MDVP:Flo(Hz)": (0.0, None),

        # Jitter/Shimmer and related measures are non-negative
        "MDVP:Jitter(%)": (0.0, None),
        "MDVP:Jitter(Abs)": (0.0, None),
        "MDVP:RAP": (0.0, None),
        "MDVP:PPQ": (0.0, None),
        "Jitter:DDP": (0.0, None),

        "MDVP:Shimmer": (0.0, None),
        "MDVP:Shimmer(dB)": (0.0, None),
        "Shimmer:APQ3": (0.0, None),
        "Shimmer:APQ5": (0.0, None),
        "MDVP:APQ": (0.0, None),
        "Shimmer:DDA": (0.0, None),

        # Noise/tonal ratios: NHR non-negative, HNR positive in practice
        "NHR": (0.0, None),
        "HNR": (0.0, None),

        # Complexity/scaling measures: typically within [0,1] for these features
        "RPDE": (0.0, 1.0),
        "DFA": (0.0, 1.0),

        # spread2 in this dataset is non-negative
        "spread2": (0.0, None),

        # D2 positive; PPE non-negative
        "D2": (0.0, None),
        "PPE": (0.0, None),
    }


def _apply_domain_validation(
    df: pd.DataFrame,
    numeric_cols: List[str],
    rules: Dict[str, Tuple[Optional[float], Optional[float]]],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Replace domain-violating values with NaN.

    Returns
    -------
    new_df, violations_by_col
    """
    work = df.copy(deep=True)
    violations: Dict[str, int] = {}

    for col in numeric_cols:
        if col not in rules:
            continue
        lo, hi = rules[col]
        s = work[col]

        mask = pd.Series(False, index=s.index)
        if lo is not None:
            mask |= s < lo
        if hi is not None:
            mask |= s > hi

        cnt = int(mask.sum())
        if cnt:
            work.loc[mask, col] = pd.NA
            violations[col] = cnt

    return work, violations


def _iqr_fences(s: pd.Series, *, k: float) -> Tuple[Optional[float], Optional[float]]:
    """Compute Tukey IQR fences for a numeric series."""
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return None, None
    return float(q1 - k * iqr), float(q3 + k * iqr)


def _winsorize_iqr(
    df: pd.DataFrame,
    numeric_cols: List[str],
    *,
    k: float,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    """
    Identify IQR outliers and winsorize (clip) them to IQR fences.

    Returns
    -------
    new_df, outliers_before_by_col, winsorized_cells_by_col
    """
    work = df.copy(deep=True)
    outliers_before: Dict[str, int] = {}
    winsorized: Dict[str, int] = {}

    for col in numeric_cols:
        s = work[col]

        # Skip if missing/constant
        if s.dropna().nunique() <= 1:
            continue

        lo, hi = _iqr_fences(s.dropna(), k=k)
        if lo is None or hi is None:
            continue

        mask = (s < lo) | (s > hi)
        cnt = int(mask.sum())
        if cnt:
            outliers_before[col] = cnt
            # Clip (winsorize) only where not NaN
            clipped = s.clip(lower=lo, upper=hi)
            changed = int((clipped != s).sum())
            work[col] = clipped
            winsorized[col] = changed

    return work, outliers_before, winsorized


def _impute_missing_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Median-impute missing values for numeric columns."""
    work = df.copy(deep=True)
    imputed_cols: List[str] = []

    for col in numeric_cols:
        if work[col].isna().sum() == 0:
            continue
        med = work[col].median()
        work[col] = work[col].fillna(med)
        imputed_cols.append(col)

    return work, sorted(imputed_cols)


def _compute_top_correlations(corr: pd.DataFrame, *, top_n: int = 15) -> pd.DataFrame:
    """Return top-N absolute correlation pairs excluding the diagonal."""
    pairs = []
    cols = list(corr.columns)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            val = float(corr.loc[c1, c2])
            pairs.append((c1, c2, val, abs(val)))

    out = pd.DataFrame(pairs, columns=["var_1", "var_2", "corr", "abs_corr"])
    out = out.sort_values("abs_corr", ascending=False).head(top_n).drop(columns=["abs_corr"])
    return out


def _print_feature_audit(
    df: pd.DataFrame,
    *,
    id_column: str,
    target_column: str,
    numeric_features: List[str],
    categorical_cols: List[str],
    max_unique_list: int,
) -> None:
    print("4) Feature inspection")
    print("-" * 74)

    # Numeric ranges
    print(f"Numeric features: {len(numeric_features)}")
    print(_numeric_range_block(df, numeric_features))
    print()

    # Categorical / nominal
    print(f"Categorical / nominal columns: {len(categorical_cols)}")
    for col in categorical_cols:
        uniq = df[col].dropna().unique().tolist()
        uniq_sorted = sorted(uniq) if col == target_column else uniq  # target sorted, ID not
        print(f"  - {col}: unique={len(uniq_sorted)}")
        # Avoid dumping huge ID lists
        print(f"    values: {_preview_unique(uniq_sorted, max_items=max_unique_list)}")
    print()


def _print_outlier_report(summary: OutlierHandlingSummary, *, k: float) -> None:
    print("5) Outliers / errors")
    print("-" * 74)
    print("How issues are defined:")
    print("  • Errors: domain violations (e.g., negative frequency, RPDE outside [0,1]).")
    print("           These are treated as invalid -> set to NaN -> median-imputed.")
    print(f"  • Outliers: Tukey IQR rule with k={k:g} (values outside [Q1 - k*IQR, Q3 + k*IQR]).")
    print("             These are winsorized (clipped) to the IQR fences.")
    print()

    if summary.domain_violations:
        print("Domain violations detected (values replaced with NaN):")
        for col, cnt in sorted(summary.domain_violations.items(), key=lambda x: (-x[1], x[0])):
            print(f"  - {col:<20} {cnt}")
    else:
        print("Domain violations detected: none")
    print()

    if summary.iqr_outliers_before:
        total_out = sum(summary.iqr_outliers_before.values())
        print(f"IQR outliers detected (before winsorization): {total_out} cells")
        # Print only the most affected columns for readability
        for col, cnt in sorted(summary.iqr_outliers_before.items(), key=lambda x: (-x[1], x[0]))[:20]:
            print(f"  - {col:<20} {cnt}")
    else:
        print("IQR outliers detected: none")
    print()

    if summary.iqr_winsorized_cells:
        total_w = sum(summary.iqr_winsorized_cells.values())
        print(f"Winsorized (clipped) cells: {total_w}")
        for col, cnt in sorted(summary.iqr_winsorized_cells.items(), key=lambda x: (-x[1], x[0]))[:20]:
            print(f"  - {col:<20} {cnt}")
    else:
        print("Winsorization applied: none")
    print()

    if summary.imputed_numeric_columns:
        print(f"Numeric columns median-imputed due to invalid/missing values: {len(summary.imputed_numeric_columns)}")
        for col in summary.imputed_numeric_columns:
            print(f"  - {col}")
    else:
        print("Median imputation: not needed")
    print()


def _print_correlation_section(
    df: pd.DataFrame,
    numeric_features: List[str],
    *,
    top_n_pairs: int,
    corr_round: int,
) -> pd.DataFrame:
    print("6) Correlation matrix (numeric features)")
    print("-" * 74)
    corr = df[numeric_features].corr(method="pearson").round(corr_round)

    # Full matrix (this is what the task asks for).
    # It is wide, but in console it's still the most explicit.
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)

    print(corr.to_string())
    print()

    # Also show top correlated pairs for interpretability
    top_pairs = _compute_top_correlations(corr, top_n=top_n_pairs)
    print(f"Top {top_n_pairs} correlated feature pairs (by |corr|):")
    print(top_pairs.to_string(index=False))
    print()

    return corr


def run(
    input_csv: str | Path,
    *,
    id_column: str = "name",
    target_column: str = "status",
    output_csv: Optional[str | Path] = None,
    corr_output_csv: Optional[str | Path] = None,
    iqr_k: float = 1.5,
    max_unique_list: int = 20,
    top_corr_pairs: int = 15,
    corr_round: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, OutlierHandlingSummary]:
    """
    Orchestrator-friendly entrypoint.

    Returns
    -------
    cleaned_df:
        Dataset after error handling + winsorization (row count preserved).
    corr:
        Pearson correlation matrix (numeric features).
    summary:
        Outlier handling summary.
    """
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    if id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' not found. Available: {list(df.columns)}")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Available: {list(df.columns)}")

    # Identify numeric features (exclude target) and categorical columns
    feature_cols = [c for c in df.columns if c not in {id_column, target_column}]
    numeric_features = [c for c in feature_cols if _is_numeric(df[c])]
    categorical_cols = [id_column, target_column] + [c for c in feature_cols if not _is_numeric(df[c])]

    # Ensure ID and target are treated as nominal for inspection
    if id_column not in categorical_cols:
        categorical_cols.insert(0, id_column)
    if target_column not in categorical_cols:
        categorical_cols.insert(1, target_column)

    print("=" * 74)
    print("Stage 3: Feature inspection, outliers/errors, correlation")
    print("=" * 74)
    print(f"Input: {str(input_csv)}")
    print(f"Rows: {len(df):,}   Columns: {df.shape[1]:,}")
    print()

    # Task 4 — feature inspection
    _print_feature_audit(
        df,
        id_column=id_column,
        target_column=target_column,
        numeric_features=numeric_features,
        categorical_cols=categorical_cols,
        max_unique_list=max_unique_list,
    )

    # Task 5 — errors/outliers
    domain_rules = _build_domain_rules()
    after_domain, domain_violations = _apply_domain_validation(df, numeric_features, domain_rules)

    after_winsor, iqr_outliers_before, winsorized = _winsorize_iqr(after_domain, numeric_features, k=iqr_k)

    after_impute, imputed_numeric = _impute_missing_numeric(after_winsor, numeric_features)

    summary = OutlierHandlingSummary(
        numeric_features=sorted(numeric_features),
        categorical_columns=sorted(set(categorical_cols)),
        domain_violations=domain_violations,
        iqr_outliers_before=iqr_outliers_before,
        iqr_winsorized_cells=winsorized,
        imputed_numeric_columns=imputed_numeric,
    )

    _print_outlier_report(summary, k=iqr_k)

    # Task 6 — correlation matrix (on cleaned numeric features)
    corr = _print_correlation_section(
        after_impute,
        numeric_features=numeric_features,
        top_n_pairs=top_corr_pairs,
        corr_round=corr_round,
    )

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        after_impute.to_csv(output_csv, index=False)
        LOGGER.info("Saved cleaned dataset to: %s", output_csv)

    if corr_output_csv is not None:
        corr_output_csv = Path(corr_output_csv)
        corr_output_csv.parent.mkdir(parents=True, exist_ok=True)
        corr.to_csv(corr_output_csv, index=True)
        LOGGER.info("Saved correlation matrix to: %s", corr_output_csv)

    return after_impute, corr, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 3: Feature inspection, outliers/errors handling, correlation matrix."
    )
    parser.add_argument("--input", required=True, help="Path to dataset CSV (recommended: dataset/parkinsons_typed.csv).")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write cleaned dataset CSV (e.g., dataset/parkinsons_stage3.csv).",
    )
    parser.add_argument(
        "--corr-output",
        default=None,
        help="Optional path to write correlation matrix CSV (e.g., reports/corr_stage3.csv).",
    )
    parser.add_argument("--id-column", default="name", help="ID column name (default: name).")
    parser.add_argument("--target-column", default="status", help="Target column name (default: status).")
    parser.add_argument("--iqr-k", type=float, default=1.5, help="Tukey IQR multiplier (default: 1.5).")
    parser.add_argument(
        "--max-unique-list",
        type=int,
        default=20,
        help="Max number of unique categorical values to print fully (default: 20).",
    )
    parser.add_argument(
        "--top-corr-pairs",
        type=int,
        default=15,
        help="How many top correlated feature pairs to print (default: 15).",
    )
    parser.add_argument(
        "--corr-round",
        type=int,
        default=3,
        help="Rounding for correlation matrix printing (default: 3 decimals).",
    )
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
        corr_output_csv=args.corr_output,
        iqr_k=args.iqr_k,
        max_unique_list=args.max_unique_list,
        top_corr_pairs=args.top_corr_pairs,
        corr_round=args.corr_round,
    )


if __name__ == "__main__":
    main()
