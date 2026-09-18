"""Deterministic dataframe analysis and chart tools used by the agent."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return name[:80] or "column"


def _column_error(column: str, df: pd.DataFrame) -> dict:
    return {"error": f"Column '{column}' not found. Available: {list(df.columns)}"}


def profile_dataset(df: pd.DataFrame) -> dict:
    return {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
        "preview": df.head(5).to_dict(orient="records"),
    }


def missingness_report(df: pd.DataFrame) -> dict:
    missing = df.isna().sum()
    missing = missing[missing > 0]
    return {
        "columns_with_missing_data": [
            {
                "column": col,
                "missing_count": int(count),
                "missing_pct": round(float(count) / len(df) * 100, 1),
            }
            for col, count in missing.items()
        ],
        "total_columns_affected": int(len(missing)),
    }


def compute_summary_stats(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        return _column_error(column, df)
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return {"error": f"Column '{column}' has no numeric data."}
    return {
        "column": column,
        "count": int(series.count()),
        "mean": round(float(series.mean()), 2),
        "median": round(float(series.median()), 2),
        "std": round(float(series.std()), 2),
        "min": round(float(series.min()), 2),
        "max": round(float(series.max()), 2),
        "q25": round(float(series.quantile(0.25)), 2),
        "q75": round(float(series.quantile(0.75)), 2),
    }


def detect_outliers(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        return _column_error(column, df)
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return {"error": f"Column '{column}' has no numeric data."}

    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    return {
        "column": column,
        "method": "IQR (1.5x)",
        "lower_bound": round(float(lower), 2),
        "upper_bound": round(float(upper), 2),
        "n_outliers": int(len(outliers)),
        "outlier_pct": round(len(outliers) / len(series) * 100, 1),
        "outlier_values": [round(float(v), 2) for v in outliers.head(10)],
    }


def compute_correlation(df: pd.DataFrame, column_a: str, column_b: str) -> dict:
    for col in (column_a, column_b):
        if col not in df.columns:
            return _column_error(col, df)

    a = pd.to_numeric(df[column_a], errors="coerce")
    b = pd.to_numeric(df[column_b], errors="coerce")
    valid = pd.concat([a, b], axis=1).dropna()
    if len(valid) < 3:
        return {"error": "Not enough overlapping numeric data to compute correlation."}

    corr = valid[column_a].corr(valid[column_b])
    if pd.isna(corr):
        return {"error": "Correlation is undefined because one of the columns has no variance."}
    return {
        "column_a": column_a,
        "column_b": column_b,
        "correlation": round(float(corr), 3),
        "n_observations": len(valid),
        "interpretation": _interpret_correlation(float(corr)),
    }


def _interpret_correlation(r: float) -> str:
    magnitude = abs(r)
    if magnitude < 0.1:
        return "negligible"
    direction = "positive" if r > 0 else "negative"
    if magnitude < 0.3:
        return f"weak {direction}"
    if magnitude < 0.5:
        return f"moderate {direction}"
    if magnitude < 0.7:
        return f"strong {direction}"
    return f"very strong {direction}"


def group_by_aggregate(df: pd.DataFrame, group_column: str, value_column: str, agg: str = "mean") -> dict:
    if group_column not in df.columns:
        return _column_error(group_column, df)
    if value_column not in df.columns:
        return _column_error(value_column, df)
    if agg not in {"mean", "sum", "median", "count", "min", "max"}:
        return {"error": f"agg must be one of mean/sum/median/count/min/max, got '{agg}'"}

    if agg == "count":
        grouped = df.groupby(group_column)[value_column].count().sort_values(ascending=False)
    else:
        series = pd.to_numeric(df[value_column], errors="coerce")
        grouped = series.groupby(df[group_column]).agg(agg).dropna().sort_values(ascending=False)

    return {
        "group_column": group_column,
        "value_column": value_column,
        "aggregation": agg,
        "results": [{"group": str(idx), "value": round(float(v), 2)} for idx, v in grouped.items()],
    }


def plot_histogram(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        return _column_error(column, df)
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return {"error": f"Column '{column}' has no numeric data."}

    _ensure_output_dir()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(series, bins=25, edgecolor="white")
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    fig.tight_layout()
    path = OUTPUT_DIR / f"histogram_{_safe_name(column)}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"chart_type": "histogram", "column": column, "file_path": str(path)}


def plot_correlation_heatmap(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return {"error": "Need at least 2 numeric columns for a correlation heatmap."}

    _ensure_output_dir()
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    path = OUTPUT_DIR / "correlation_heatmap.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"chart_type": "correlation_heatmap", "file_path": str(path), "columns": list(corr.columns)}


def plot_bar_by_group(df: pd.DataFrame, group_column: str, value_column: str, agg: str = "mean") -> dict:
    result = group_by_aggregate(df, group_column, value_column, agg)
    if "error" in result:
        return result

    _ensure_output_dir()
    labels = [r["group"] for r in result["results"]]
    values = [r["value"] for r in result["results"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, values)
    ax.set_title(f"{agg.title()} {value_column} by {group_column}")
    ax.set_ylabel(f"{agg.title()} {value_column}")
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    path = OUTPUT_DIR / f"bar_{_safe_name(value_column)}_by_{_safe_name(group_column)}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"chart_type": "bar", "file_path": str(path), **result}


TOOL_FUNCTIONS = {
    "profile_dataset": profile_dataset,
    "missingness_report": missingness_report,
    "compute_summary_stats": compute_summary_stats,
    "detect_outliers": detect_outliers,
    "compute_correlation": compute_correlation,
    "group_by_aggregate": group_by_aggregate,
    "plot_histogram": plot_histogram,
    "plot_correlation_heatmap": plot_correlation_heatmap,
    "plot_bar_by_group": plot_bar_by_group,
}


if __name__ == "__main__":
    sample = pd.read_csv(BASE_DIR / "sample_data" / "sales_data.csv")
    print(profile_dataset(sample))
    print(missingness_report(sample))
