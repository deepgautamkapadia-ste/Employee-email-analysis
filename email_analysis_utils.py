"""
Shared helpers for the employee email analysis workflow.

These utilities keep the task scripts small, consistent, and easy to adapt
when the input CSV uses different column names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"


def resolve_column_name(
    df: pd.DataFrame,
    explicit_name: Optional[str],
    candidate_names: Iterable[str],
    friendly_label: str,
) -> str:
    """
    Resolve a column name either from an explicit user-supplied value or by
    searching common candidate names.
    """

    if explicit_name:
        if explicit_name not in df.columns:
            available = ", ".join(map(str, df.columns))
            raise KeyError(
                f"{friendly_label} column '{explicit_name}' was not found. "
                f"Available columns: {available}"
            )
        return explicit_name

    for candidate in candidate_names:
        if candidate in df.columns:
            return candidate

    available = ", ".join(map(str, df.columns))
    candidates = ", ".join(candidate_names)
    raise KeyError(
        f"Could not infer {friendly_label} column. Tried: {candidates}. "
        f"Available columns: {available}"
    )


def ensure_visualizations_dir() -> Path:
    """Create the visualizations directory if it does not already exist."""

    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    return VISUALIZATIONS_DIR


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV with a friendly error if the file is missing."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path.resolve()}")
    return pd.read_csv(csv_path)


def coerce_datetime(series: pd.Series) -> pd.Series:
    """Convert a date-like series into pandas datetime, preserving NaT for bad rows."""

    return pd.to_datetime(series, errors="coerce")


def normalize_month_period(series: pd.Series) -> pd.Series:
    """Convert datetimes into a YYYY-MM monthly period label."""

    return series.dt.to_period("M").astype(str)


def safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    """Coerce a series to numeric and fill missing values with a safe default."""

    return pd.to_numeric(series, errors="coerce").fillna(default)


def combine_text_columns(*columns: pd.Series) -> pd.Series:
    """
    Combine multiple text columns into a single cleaned string.

    Empty values are skipped so the result stays readable when subject or body
    is missing.
    """

    if not columns:
        return pd.Series(dtype=str)

    combined = columns[0].fillna("").astype(str).str.strip()
    for column in columns[1:]:
        part = column.fillna("").astype(str).str.strip()
        combined = combined + "\n" + part
    return combined.str.replace(r"(\n\s*){2,}", "\n", regex=True).str.strip()


def pretty_print_section(title: str) -> None:
    """Print a compact text section header for console output."""

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
