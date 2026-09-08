"""Run basic, univariate, bivariate, multivariate, and missing-value analysis.

Usage:
    python src/visualization/exploratory_data_analysis.py
    python src/visualization/exploratory_data_analysis.py --input data/raw/players_stat.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/processed/player_match_dataset_cleaned.csv"
DEFAULT_OUTPUT = ROOT / "reports/eda"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--pairplot-sample",
        type=int,
        default=1500,
        help="Maximum rows used for the multivariate pairplot.",
    )
    return parser.parse_args()


def save_basic_inspection(df: pd.DataFrame, output_dir: Path) -> None:
    inspection = pd.DataFrame(
        {
            "rows": [len(df)],
            "columns": [df.shape[1]],
            "duplicate_rows": [df.duplicated().sum()],
            "memory_mb": [df.memory_usage(deep=True).sum() / 1_000_000],
        }
    )
    inspection.to_csv(output_dir / "basic_inspection.csv", index=False)
    df.dtypes.rename("dtype").to_frame().to_csv(output_dir / "data_types.csv")
    df.describe(include="all").T.to_csv(
        output_dir / "descriptive_statistics.csv"
    )


def save_missing_value_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    missing = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_percent": df.isna().mean().mul(100),
        }
    ).sort_values("missing_percent", ascending=False)
    missing.to_csv(output_dir / "missing_values.csv")

    nonzero_missing = missing[missing["missing_count"] > 0]
    if nonzero_missing.empty:
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=nonzero_missing.reset_index(),
        x="missing_percent",
        y="index",
        color="#d95f02",
    )
    plt.xlabel("Missing values (%)")
    plt.ylabel("Column")
    plt.title("Missing-value analysis")
    plt.tight_layout()
    plt.savefig(output_dir / "missing_values.png", dpi=150)
    plt.close()


def save_missing_value_strategy_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    missing_mask = df.isna()
    missing_per_row = missing_mask.sum(axis=1)
    missing_per_column = missing_mask.sum()
    row_count, column_count = df.shape

    row_summary = pd.DataFrame(
        {
            "missing_values_in_row": missing_per_row.value_counts().sort_index().index,
            "row_count": missing_per_row.value_counts().sort_index().values,
        }
    )
    row_summary["row_percent"] = row_summary["row_count"].div(row_count).mul(100)
    row_summary.to_csv(output_dir / "missing_values_per_row.csv", index=False)

    incomplete_rows = int((missing_per_row > 0).sum())
    complete_rows = int((missing_per_row == 0).sum())
    affected_columns = int((missing_per_column > 0).sum())
    strategy_summary = pd.DataFrame(
        [
            {
                "strategy": "Drop rows with any missing value",
                "rows_deleted": incomplete_rows,
                "rows_deleted_percent": incomplete_rows / row_count * 100,
                "rows_retained": complete_rows,
                "rows_retained_percent": complete_rows / row_count * 100,
                "columns_deleted": 0,
                "columns_retained_percent": 100.0,
            },
            {
                "strategy": "Drop every column containing missing values",
                "rows_deleted": 0,
                "rows_deleted_percent": 0.0,
                "rows_retained": row_count,
                "rows_retained_percent": 100.0,
                "columns_deleted": affected_columns,
                "columns_retained_percent": (column_count - affected_columns)
                / column_count
                * 100,
            },
        ]
    )
    strategy_summary.to_csv(output_dir / "missing_value_strategies.csv", index=False)

    patterns = missing_mask.loc[missing_per_row > 0].astype(int)
    patterns["rows"] = 1
    patterns = (
        patterns.groupby(list(missing_mask.columns), dropna=False)["rows"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    patterns["row_percent_of_incomplete"] = patterns["rows"].div(incomplete_rows).mul(100)
    patterns.to_csv(output_dir / "missing_value_patterns.csv", index=False)

    per_column_loss = pd.DataFrame(
        {
            "missing_count": missing_per_column,
            "missing_percent_of_rows": missing_per_column / row_count * 100,
            "rows_deleted_if_column_is_required": missing_per_column,
            "rows_retained_if_column_is_required": row_count - missing_per_column,
        }
    ).sort_values("missing_count", ascending=False)
    per_column_loss.to_csv(output_dir / "missing_value_column_deletion_cost.csv")

    print("Missing-value deletion analysis:")
    print(f"  Missing cells: {int(missing_mask.values.sum())} ({missing_mask.values.mean() * 100:.4f}%)")
    print(f"  Incomplete rows: {incomplete_rows} ({incomplete_rows / row_count * 100:.4f}%)")
    print(f"  Complete-case retention: {complete_rows / row_count * 100:.4f}%")
    print(f"  Affected columns: {affected_columns}/{column_count}")


def save_univariate_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    numeric = df.select_dtypes(include="number")
    categorical = df.select_dtypes(exclude="number")

    numeric.describe().T.to_csv(output_dir / "numeric_summary.csv")
    categorical_summary = pd.DataFrame(
        {
            "unique_values": categorical.nunique(dropna=False),
            "most_common": categorical.mode(dropna=False).iloc[0],
        }
    )
    categorical_summary.to_csv(output_dir / "categorical_summary.csv")

    if numeric.empty:
        return

    numeric.hist(figsize=(16, 12), bins=30, edgecolor="white")
    plt.suptitle("Univariate distributions", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "numeric_distributions.png", dpi=150)
    plt.close()


def save_bivariate_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return

    correlation = numeric.corr()
    correlation.to_csv(output_dir / "numeric_correlations.csv")

    target = "rating2_all"
    if target in numeric.columns:
        target_correlation = (
            correlation[target]
            .drop(target)
            .sort_values(key=lambda values: values.abs(), ascending=False)
            .rename("correlation_with_rating2_all")
        )
        target_correlation.to_csv(output_dir / "target_correlations.csv")

        strongest_feature = target_correlation.index[0] if not target_correlation.empty else None
        if strongest_feature:
            sample = df[[strongest_feature, target]].dropna()
            if len(sample) > 5000:
                sample = sample.sample(5000, random_state=42)
            plt.figure(figsize=(8, 6))
            sns.regplot(data=sample, x=strongest_feature, y=target, scatter_kws={"alpha": 0.35})
            plt.title(f"{target} vs {strongest_feature}")
            plt.tight_layout()
            plt.savefig(output_dir / "target_vs_strongest_feature.png", dpi=150)
            plt.close()

    for category in ("agent", "map", "team"):
        if category not in df.columns or target not in df.columns:
            continue
        grouped = (
            df.groupby(category, dropna=False)[target]
            .agg(["count", "mean", "median"])
            .sort_values("mean", ascending=False)
        )
        grouped.to_csv(output_dir / f"{category}_vs_{target}.csv")


def save_multivariate_analysis(
    df: pd.DataFrame, output_dir: Path, pairplot_sample: int
) -> None:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return

    plt.figure(figsize=(14, 11))
    sns.heatmap(numeric.corr(), cmap="vlag", center=0, linewidths=0.2)
    plt.title("Multivariate numeric correlation matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close()

    preferred = [
        column
        for column in ["rating2_all", "acs_all", "kills_all", "deaths_all", "assists_all"]
        if column in numeric.columns
    ]
    if len(preferred) < 2:
        return

    sample = df[preferred].dropna()
    if len(sample) > pairplot_sample:
        sample = sample.sample(pairplot_sample, random_state=42)
    grid = sns.pairplot(sample, corner=True, plot_kws={"alpha": 0.35, "s": 18})
    grid.figure.suptitle("Multivariate relationships", y=1.02)
    grid.figure.savefig(output_dir / "pairplot.png", dpi=150)
    plt.close(grid.figure)


def run_analysis(input_path: Path, output_dir: Path, pairplot_sample: int) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path, low_memory=False)

    save_basic_inspection(df, output_dir)
    save_univariate_analysis(df, output_dir)
    save_bivariate_analysis(df, output_dir)
    save_multivariate_analysis(df, output_dir, pairplot_sample)
    save_missing_value_analysis(df, output_dir)
    save_missing_value_strategy_analysis(df, output_dir)

    print(f"Analysed {len(df):,} rows and {len(df.columns):,} columns")
    print(f"EDA outputs saved to: {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    sns.set_theme(style="whitegrid")
    run_analysis(args.input, args.output, args.pairplot_sample)