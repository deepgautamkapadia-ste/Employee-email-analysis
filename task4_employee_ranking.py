"""
Task 4: Employee ranking by month.

For each month, this script extracts:
    1. Top Three Positive Employees
    2. Top Three Negative Employees

Ties are broken alphabetically by employee name/identifier.
The results are printed to the terminal and saved as visual summaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from email_analysis_utils import ensure_visualizations_dir, load_csv, pretty_print_section, resolve_column_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank employees by monthly score.")
    parser.add_argument("--input", default="monthly_scores.csv", help="Input monthly CSV path.")
    parser.add_argument(
        "--employee-column",
        default=None,
        help="Employee identifier column name, if not one of the default candidates.",
    )
    parser.add_argument(
        "--month-column",
        default=None,
        help="Month column name, if not one of the default candidates.",
    )
    parser.add_argument(
        "--score-column",
        default=None,
        help="Monthly score column name, if not one of the default candidates.",
    )
    return parser.parse_args()


def format_table(df: pd.DataFrame, title: str) -> str:
    if df.empty:
        return f"{title}\n  No rows available."
    return f"{title}\n{df.to_string(index=False)}"


def save_month_visualization(
    month_label: str,
    positive_df: pd.DataFrame,
    negative_df: pd.DataFrame,
    employee_column: str,
    score_column: str,
    viz_dir: Path,
) -> Path:
    """
    Save a horizontal bar chart showing the top positive and negative employees
    for one month.
    """

    safe_month = month_label.replace("/", "_").replace(" ", "_")
    output_path = viz_dir / f"employee_rankings_{safe_month}.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    def _plot(ax, frame: pd.DataFrame, title: str, color: str, invert_axis: bool = False) -> None:
        if frame.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
            ax.set_axis_off()
            return
        plot_df = frame.sort_values(score_column, ascending=invert_axis)
        ax.barh(plot_df[employee_column].astype(str), plot_df[score_column], color=color)
        ax.set_title(title)
        ax.set_xlabel(score_column)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

    _plot(axes[0], positive_df, f"{month_label} - Top Positive", "#2ca02c", invert_axis=False)
    _plot(axes[1], negative_df, f"{month_label} - Top Negative", "#d62728", invert_axis=True)

    fig.suptitle(f"Employee Rankings for {month_label}", fontsize=14)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    pretty_print_section("Task 4: Employee Ranking")

    df = load_csv(args.input)
    employee_column = resolve_column_name(
        df,
        args.employee_column,
        ["from", "From", "employee", "Employee", "Employee_ID", "EmployeeID", "Employee Name", "Name"],
        "employee",
    )
    month_column = resolve_column_name(
        df,
        args.month_column,
        ["Month_Year", "Month", "MonthYear", "Period"],
        "month",
    )
    score_column = resolve_column_name(
        df,
        args.score_column,
        ["Monthly_Score", "Cumulative_Monthly_Score", "Score"],
        "score",
    )

    viz_dir = ensure_visualizations_dir()

    for month_label, month_df in df.groupby(month_column, sort=True):
        month_df = month_df.copy()

        positive_rank = month_df.sort_values(
            by=[score_column, employee_column], ascending=[False, True]
        ).head(3)
        negative_rank = month_df.sort_values(
            by=[score_column, employee_column], ascending=[True, True]
        ).head(3)

        pretty_print_section(f"Month: {month_label}")
        print(format_table(positive_rank[[employee_column, score_column]], "Top Three Positive Employees"))
        print()
        print(format_table(negative_rank[[employee_column, score_column]], "Top Three Negative Employees"))

        saved_plot = save_month_visualization(
            month_label=str(month_label),
            positive_df=positive_rank,
            negative_df=negative_rank,
            employee_column=employee_column,
            score_column=score_column,
            viz_dir=viz_dir,
        )
        print(f"\nSaved ranking visualization: {saved_plot}")


if __name__ == "__main__":
    main()
