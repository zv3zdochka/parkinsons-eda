# 4_visualize_data.py
"""
Stage 4 — Data visualization (matplotlib + optional seaborn).

What was changed (relative to your current version)
---------------------------------------------------
The `pairplot_top_features.png` was reworked to be more readable and compact:
- switched from `sns.pairplot(...)` to an explicit `sns.PairGrid(...)` so we can:
  * control legend placement (inside the figure, no huge empty right margin),
  * control subplot sizing consistently,
  * keep the same "corner" layout but with tighter spacing,
  * use consistent scatter/hist settings across seaborn versions.

Everything else (bars, hist pages, boxplots, scatter-vs-target, heatmap, console report)
is kept compatible with the existing pipeline and CLI.

Outputs
-------
Images are saved to an output directory (default: reports/stage4_figures):
  - bar_status.png
  - bar_recordings_per_subject.png
  - hist_numeric_page_*.png
  - box_top_features_by_status.png
  - scatter_top_features_vs_target.png
  - pairplot_top_features.png          (requires seaborn; improved layout)
  - heatmap_corr_top_features.png      (requires seaborn; minor label tweaks)

Recommended input
-----------------
Use the cleaned dataset from Stage 3 (after winsorization):
  dataset/parkinsons_stage3.csv
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VizConfig:
    """Visualization configuration."""
    top_k_features: int
    hist_cols: int
    hist_rows: int
    corr_threshold: float
    dpi: int


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _try_import_seaborn():
    try:
        import seaborn as sns  # type: ignore
        return sns
    except Exception:
        return None


def _ensure_roles(df: pd.DataFrame, *, id_column: str, target_column: str) -> pd.DataFrame:
    if id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' not found. Available: {list(df.columns)}")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Available: {list(df.columns)}")

    out = df.copy(deep=True)

    # Normalize ID and target to stable types for plotting and grouping
    out[id_column] = out[id_column].astype("string")

    # Target should be 0/1; enforce int if possible
    if not pd.api.types.is_numeric_dtype(out[target_column].dtype):
        out[target_column] = pd.to_numeric(out[target_column], errors="coerce")
    out = out.dropna(subset=[id_column, target_column])
    out[target_column] = out[target_column].astype(int)

    uniq = set(out[target_column].unique().tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(
            f"Unexpected target values in '{target_column}': {sorted(uniq)}. Expected binary labels {0,1}."
        )
    return out


def _subject_id_from_name(name: pd.Series) -> pd.Series:
    """
    Extract a subject identifier from 'name'.

    Typical format: phon_R01_S01_1
    Subject id:     phon_R01_S01  (everything except the trailing recording index).
    """
    return name.astype("string").str.rsplit("_", n=1).str[0]


def _numeric_feature_columns(df: pd.DataFrame, *, id_column: str, target_column: str) -> List[str]:
    feature_cols = [c for c in df.columns if c not in {id_column, target_column}]
    numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c].dtype)]
    return numeric


def _target_balance(df: pd.DataFrame, target_column: str) -> Tuple[pd.Series, pd.Series]:
    counts = df[target_column].value_counts().sort_index()
    pct = (counts / len(df) * 100.0).round(2)
    return counts, pct


def _top_features_by_target_corr(df: pd.DataFrame, numeric_cols: List[str], target_column: str, top_k: int) -> pd.DataFrame:
    """
    For a binary 0/1 target, Pearson correlation with the target equals point-biserial correlation.
    This is a quick association scan (not a causal claim).
    """
    rows = []
    y = df[target_column]
    for col in numeric_cols:
        corr = float(df[col].corr(y))
        rows.append((col, corr, abs(corr)))

    out = pd.DataFrame(rows, columns=["feature", "corr_with_target", "abs_corr"])
    out = out.sort_values("abs_corr", ascending=False).drop(columns=["abs_corr"]).head(top_k)
    return out


def _save_bar_status(df: pd.DataFrame, target_column: str, out_dir: Path, *, dpi: int) -> Path:
    counts, pct = _target_balance(df, target_column)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(k) for k in counts.index], counts.values)
    ax.set_title("Target distribution (status)")
    ax.set_xlabel("status (0=healthy, 1=PD)")
    ax.set_ylabel("count")

    for i, k in enumerate(counts.index):
        ax.text(i, counts.loc[k], f"{counts.loc[k]} ({pct.loc[k]}%)", ha="center", va="bottom")

    fig.tight_layout()
    path = out_dir / "bar_status.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _save_bar_recordings_per_subject(df: pd.DataFrame, id_column: str, out_dir: Path, *, dpi: int) -> Path:
    subject_id = _subject_id_from_name(df[id_column])
    counts = subject_id.value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(counts)), counts.values)
    ax.set_title("Recordings per subject (derived from `name`)")
    ax.set_xlabel("subjects (sorted)")
    ax.set_ylabel("number of recordings")

    ax.text(
        0.99,
        0.95,
        f"subjects: {len(counts)}\nmin: {int(counts.min())}  median: {float(counts.median()):.1f}  max: {int(counts.max())}",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )

    fig.tight_layout()
    path = out_dir / "bar_recordings_per_subject.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _save_histograms(df: pd.DataFrame, numeric_cols: List[str], out_dir: Path, *, rows: int, cols: int, dpi: int) -> List[Path]:
    """Save histograms for all numeric columns as multiple "pages" (grid images)."""
    per_page = rows * cols
    paths: List[Path] = []

    for page_start in range(0, len(numeric_cols), per_page):
        page_cols = numeric_cols[page_start: page_start + per_page]

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 3.0))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for ax, col in zip(axes, page_cols):
            ax.hist(df[col].dropna().values, bins=30)
            ax.set_title(col, fontsize=10)
            ax.set_xlabel("value")
            ax.set_ylabel("count")

        for ax in axes[len(page_cols):]:
            ax.axis("off")

        fig.suptitle("Histograms of numeric variables", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        page_idx = page_start // per_page + 1
        path = out_dir / f"hist_numeric_page_{page_idx:02d}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        paths.append(path)

    return paths


def _save_boxplots_top_features(df: pd.DataFrame, top_features: List[str], target_column: str, out_dir: Path, *, dpi: int) -> Path:
    """Boxplots of selected features split by status."""
    n = len(top_features)
    fig, axes = plt.subplots(n, 1, figsize=(9, max(3, 2.2 * n)))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, top_features):
        data0 = df.loc[df[target_column] == 0, col].dropna()
        data1 = df.loc[df[target_column] == 1, col].dropna()
        ax.boxplot([data0.values, data1.values], labels=["status=0", "status=1"])
        ax.set_title(col)
        ax.set_ylabel("value")

    fig.suptitle("Top features by association with target (boxplots)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / "box_top_features_by_status.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _save_scatter_top_features_vs_target(df: pd.DataFrame, top_features: List[str], target_column: str, out_dir: Path, *, dpi: int) -> Path:
    """Scatter: feature value vs target (0/1) with jitter on y for readability."""
    n = len(top_features)
    fig, axes = plt.subplots(n, 1, figsize=(9, max(3, 2.2 * n)))
    if n == 1:
        axes = [axes]

    rng = np.random.default_rng(42)

    for ax, col in zip(axes, top_features):
        y = df[target_column].astype(float).values
        y_jitter = y + rng.normal(0, 0.03, size=len(y))

        ax.scatter(df[col].values, y_jitter, s=14, alpha=0.7)
        ax.set_title(f"{col} vs {target_column}")
        ax.set_xlabel(col)
        ax.set_ylabel(target_column)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["0", "1"])

    fig.suptitle("Top features vs target (scatter with jitter)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / "scatter_top_features_vs_target.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _save_pairplot_compact_and_heatmap_seaborn(
    df: pd.DataFrame,
    top_features: List[str],
    target_column: str,
    out_dir: Path,
    *,
    dpi: int,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Compact pairwise plot + correlation heatmap for selected features.

    Why PairGrid instead of seaborn.pairplot?
    - Pairplot tends to place the legend outside, creating a large blank area.
    - PairGrid gives explicit control over legend position and spacing.
    """
    sns = _try_import_seaborn()
    if sns is None:
        return None, None

    # Prepare data for seaborn
    pair_df = df[top_features + [target_column]].copy()
    pair_df[target_column] = pair_df[target_column].astype("category")

    # --- PairGrid (compact, legend inside) ---
    pair_path = out_dir / "pairplot_top_features.png"

    grid = sns.PairGrid(
        data=pair_df,
        vars=top_features,
        hue=target_column,
        corner=True,
        height=2.25,
        diag_sharey=False,
    )

    # Lower triangle: scatter
    grid.map_lower(
        sns.scatterplot,
        s=18,
        alpha=0.75,
        linewidth=0,
    )

    # Diagonal: histogram
    grid.map_diag(
        sns.histplot,
        bins=18,
        alpha=0.75,
    )

    # Legend: keep it *inside* the figure (prevents wide blank margins)
    grid.add_legend(title=target_column)
    if grid._legend is not None:
        grid._legend.set_bbox_to_anchor((0.98, 0.98))
        grid._legend.set_loc("upper right")
        grid._legend.get_frame().set_alpha(0.9)

    grid.fig.suptitle("Pairwise relationships (top features)", y=1.02)

    # Tighten layout without pushing legend outside
    grid.fig.subplots_adjust(top=0.93, right=0.98)
    grid.fig.savefig(pair_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(grid.fig)

    # --- Heatmap for the same subset ---
    heat_path = out_dir / "heatmap_corr_top_features.png"
    corr = df[top_features].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, ax=ax, annot=False, square=True, cbar=True)
    ax.set_title("Correlation heatmap (top features)")

    # Improve readability of tick labels a bit
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()
    fig.savefig(heat_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    return pair_path, heat_path


def _count_high_corr_pairs(df: pd.DataFrame, numeric_cols: List[str], *, threshold: float) -> int:
    """Count feature pairs with |corr| >= threshold (upper triangle only, excluding diagonal)."""
    corr = df[numeric_cols].corr().abs()
    upper = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    vals = corr.where(upper).stack()
    return int((vals >= threshold).sum())


def _print_intermediate_conclusions(
    df: pd.DataFrame,
    *,
    numeric_cols: List[str],
    target_column: str,
    top_corr_df: pd.DataFrame,
    corr_threshold: float,
) -> None:
    counts, pct = _target_balance(df, target_column)

    skew = df[numeric_cols].skew(numeric_only=True).sort_values(key=lambda s: s.abs(), ascending=False)
    skew_top = skew.head(6)

    try:
        high_corr_pairs = _count_high_corr_pairs(df, numeric_cols, threshold=corr_threshold)
    except Exception:
        high_corr_pairs = -1

    print("Intermediate conclusions (Stage 4)")
    print("-" * 74)
    print("Class balance:")
    for k in counts.index:
        print(f"  - {target_column}={int(k)}: {int(counts.loc[k])} rows ({float(pct.loc[k]):.2f}%)")
    print()

    print("Most target-associated features (Pearson corr with binary target):")
    for _, row in top_corr_df.iterrows():
        print(f"  - {row['feature']:<16} corr={row['corr_with_target']:+.3f}")
    print()

    print("Distribution shape (largest |skew| among numeric features):")
    for col, val in skew_top.items():
        print(f"  - {col:<16} skew={float(val):+.3f}")
    print()

    if high_corr_pairs >= 0:
        print(f"Multicollinearity signal: number of feature pairs with |corr| >= {corr_threshold:.2f} is {high_corr_pairs}.")
        print("Interpretation: many jitter/shimmer-derived measures are strongly redundant; feature selection or regularization may help.")
    else:
        print("Multicollinearity signal: not computed (environment limitation).")
    print()

    print("Practical notes:")
    print("  • Several variables show long-tailed behavior (expected for biomedical signal features).")
    print("  • Pairwise plots (top features) help visually assess separation between status=0 and status=1.")
    print("  • Remember: multiple recordings per subject exist; evaluation later must be group-aware to avoid leakage.")
    print("-" * 74)
    print()


def run(
    input_csv: str | Path,
    *,
    id_column: str = "name",
    target_column: str = "status",
    output_dir: str | Path = "reports/stage4_figures",
    config: VizConfig = VizConfig(top_k_features=6, hist_cols=4, hist_rows=4, corr_threshold=0.95, dpi=160),
) -> None:
    """Orchestrator-friendly entrypoint: generates plots and prints a concise EDA summary."""
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"CSV not found: {input_csv}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    df = _ensure_roles(df, id_column=id_column, target_column=target_column)

    numeric_cols = _numeric_feature_columns(df, id_column=id_column, target_column=target_column)
    if not numeric_cols:
        raise ValueError("No numeric feature columns found; cannot create histograms/correlations.")

    top_corr = _top_features_by_target_corr(df, numeric_cols, target_column, top_k=config.top_k_features)
    top_features = top_corr["feature"].tolist()

    paths: List[Path] = []
    paths.append(_save_bar_status(df, target_column, out_dir, dpi=config.dpi))
    paths.append(_save_bar_recordings_per_subject(df, id_column, out_dir, dpi=config.dpi))
    paths.extend(_save_histograms(df, numeric_cols, out_dir, rows=config.hist_rows, cols=config.hist_cols, dpi=config.dpi))
    paths.append(_save_boxplots_top_features(df, top_features, target_column, out_dir, dpi=config.dpi))
    paths.append(_save_scatter_top_features_vs_target(df, top_features, target_column, out_dir, dpi=config.dpi))

    pair_path, heat_path = _save_pairplot_compact_and_heatmap_seaborn(df, top_features, target_column, out_dir, dpi=config.dpi)
    if pair_path is not None:
        paths.append(pair_path)
    if heat_path is not None:
        paths.append(heat_path)

    print("=" * 74)
    print("Stage 4: Visualizations")
    print("=" * 74)
    print(f"Input:  {str(input_csv)}")
    print(f"Output: {str(out_dir)}")
    print()
    print("Generated files:")
    for p in paths:
        print(f"  - {p.as_posix()}")
    print()

    _print_intermediate_conclusions(
        df,
        numeric_cols=numeric_cols,
        target_column=target_column,
        top_corr_df=top_corr,
        corr_threshold=config.corr_threshold,
    )

    if _try_import_seaborn() is None:
        print("Note: seaborn is not available. Pairplot and heatmap were skipped.")
        print("      Install seaborn if you want those plots: pip install seaborn")
        print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4: Visualize numeric/categorical variables and relationships.")
    parser.add_argument("--input", required=True, help="Path to dataset CSV (recommended: dataset/parkinsons_stage3.csv).")
    parser.add_argument("--output-dir", default="reports/stage4_figures", help="Directory to save images.")
    parser.add_argument("--id-column", default="name", help="ID column name (default: name).")
    parser.add_argument("--target-column", default="status", help="Target column name (default: status).")
    parser.add_argument("--top-k", type=int, default=6, help="How many top features to use in pairwise plots (default: 6).")
    parser.add_argument("--hist-rows", type=int, default=4, help="Histogram grid rows per page (default: 4).")
    parser.add_argument("--hist-cols", type=int, default=4, help="Histogram grid cols per page (default: 4).")
    parser.add_argument("--corr-threshold", type=float, default=0.95, help="High-correlation threshold for summary (default: 0.95).")
    parser.add_argument("--dpi", type=int, default=160, help="Saved image DPI (default: 160).")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)

    cfg = VizConfig(
        top_k_features=args.top_k,
        hist_cols=args.hist_cols,
        hist_rows=args.hist_rows,
        corr_threshold=args.corr_threshold,
        dpi=args.dpi,
    )

    run(
        args.input,
        id_column=args.id_column,
        target_column=args.target_column,
        output_dir=args.output_dir,
        config=cfg,
    )


if __name__ == "__main__":
    main()
