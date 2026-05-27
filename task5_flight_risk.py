"""
Task 5: Flight-risk flagging.

An employee is flagged as a flight risk if they send 4 or more Negative messages
within any rolling 30-day window, computed chronologically per employee.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from email_analysis_utils import coerce_datetime, ensure_visualizations_dir, load_csv, pretty_print_section, resolve_column_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flag employees with repeated negative sentiment.")
    parser.add_argument("--input", default="labeled_data.csv", help="Input labeled CSV path.")
    parser.add_argument(
        "--employee-column",
        default=None,
        help="Employee identifier column name, if not one of the default candidates.",
    )
    parser.add_argument(
        "--date-column",
        default=None,
        help="Date/timestamp column name, if not one of the default candidates.",
    )
    parser.add_argument(
        "--sentiment-column",
        default=None,
        help="Sentiment label column name, if not one of the default candidates.",
    )
    return parser.parse_args()


def max_negative_rolling_count(employee_df: pd.DataFrame, date_column: str) -> int:
    """
    Compute the maximum number of negative messages observed in any 30-day
    rolling window for one employee.
    """

    negative_dates = employee_df.loc[employee_df["is_negative"], date_column].dropna().sort_values()
    if negative_dates.empty:
        return 0

    # Rolling windows require a DatetimeIndex. A one-valued Series lets us count
    # how many negative messages fall into each 30-day span.
    rolling_counts = pd.Series(1, index=negative_dates).rolling("30D").sum()
    return int(rolling_counts.max())


def main() -> None:
    args = parse_args()
    pretty_print_section("Task 5: Flight Risk Detection")

    df = load_csv(args.input)
    employee_column = resolve_column_name(
        df,
        args.employee_column,
        ["from", "From", "employee", "Employee", "Employee_ID", "EmployeeID", "Employee Name", "Name"],
        "employee",
    )
    date_column = resolve_column_name(
        df,
        args.date_column,
        ["date", "Date", "Timestamp", "Datetime", "Time", "Created_At"],
        "date",
    )
    sentiment_column = resolve_column_name(
        df,
        args.sentiment_column,
        ["Sentiment_Label", "Sentiment", "Label"],
        "sentiment",
    )

    df[date_column] = coerce_datetime(df[date_column])
    df = df.dropna(subset=[date_column]).copy()
    df["is_negative"] = df[sentiment_column].eq("Negative")

    employee_risk_rows = []
    for employee_id, employee_df in df.groupby(employee_column, sort=True):
        employee_df = employee_df.sort_values(date_column)
        max_count = max_negative_rolling_count(employee_df, date_column)
        employee_risk_rows.append(
            {
                employee_column: employee_id,
                "Max_Negative_30D": max_count,
                "Flagged_Flight_Risk": max_count >= 4,
            }
        )

    risk_df = pd.DataFrame(employee_risk_rows).sort_values(
        ["Flagged_Flight_Risk", "Max_Negative_30D", employee_column],
        ascending=[False, False, True],
    )

    flagged = risk_df.loc[risk_df["Flagged_Flight_Risk"], employee_column].astype(str).tolist()

    pretty_print_section("Flagged Employees")
    if flagged:
        print("\n".join(flagged))
    else:
        print("No employees met the flight-risk threshold.")

    viz_dir = ensure_visualizations_dir()
    plot_path = viz_dir / "flight_risk_summary.png"

    plt.figure(figsize=(10, 6))
    flagged_df = risk_df[risk_df["Flagged_Flight_Risk"]].copy()
    if flagged_df.empty:
        plt.text(0.5, 0.5, "No flight risks detected", ha="center", va="center", fontsize=14)
        plt.axis("off")
    else:
        plt.barh(
            flagged_df[employee_column].astype(str),
            flagged_df["Max_Negative_30D"],
            color="#d62728",
        )
        plt.axvline(4, linestyle="--", color="black", alpha=0.7, label="Threshold = 4")
        plt.title("Flagged Flight Risks: Max Negative Messages in Any 30-Day Window")
        plt.xlabel("Maximum Negative Messages in 30 Days")
        plt.ylabel("Employee")
        plt.gca().invert_yaxis()
        plt.legend()
        plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nSaved flight-risk summary plot to: {plot_path}")


if __name__ == "__main__":
    main()
