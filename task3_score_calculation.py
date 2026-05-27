"""
Task 3: Monthly sentiment score calculation.

Each labeled message is mapped to a numeric score:
    Positive -> +1
    Negative -> -1
    Neutral  -> 0

The script then aggregates by employee and calendar month to produce a monthly
score that resets at the start of each month.
"""

from __future__ import annotations

import argparse

import pandas as pd

from email_analysis_utils import (
    coerce_datetime,
    load_csv,
    normalize_month_period,
    pretty_print_section,
    resolve_column_name,
    safe_numeric,
)


SENTIMENT_TO_SCORE = {"Positive": 1, "Negative": -1, "Neutral": 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate monthly sentiment scores.")
    parser.add_argument("--input", default="labeled_data.csv", help="Input labeled CSV path.")
    parser.add_argument("--output", default="monthly_scores.csv", help="Output CSV path.")
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


def main() -> None:
    args = parse_args()
    pretty_print_section("Task 3: Monthly Score Calculation")

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
    df["Month_Year"] = normalize_month_period(df[date_column])
    df["Sentiment_Score"] = df[sentiment_column].map(SENTIMENT_TO_SCORE)
    df["Sentiment_Score"] = safe_numeric(df["Sentiment_Score"], default=0)

    monthly = (
        df.sort_values([employee_column, date_column])
        .groupby([employee_column, "Month_Year"], as_index=False)
        .agg(
            Monthly_Score=("Sentiment_Score", "sum"),
            Message_Count=("Sentiment_Score", "size"),
            Positive_Count=(sentiment_column, lambda s: (s == "Positive").sum()),
            Negative_Count=(sentiment_column, lambda s: (s == "Negative").sum()),
            Neutral_Count=(sentiment_column, lambda s: (s == "Neutral").sum()),
            First_Message_Date=(date_column, "min"),
            Last_Message_Date=(date_column, "max"),
        )
    )

    # The cumulative monthly score resets every new month, so the monthly total
    # is the reset value for that month. This explicit column is useful if later
    # scripts want to distinguish "monthly total" from other derived metrics.
    monthly["Cumulative_Monthly_Score"] = monthly["Monthly_Score"]

    monthly = monthly.sort_values([employee_column, "Month_Year"]).reset_index(drop=True)
    monthly.to_csv(args.output, index=False)

    pretty_print_section("Monthly Score Preview")
    print(monthly.head(10).to_string(index=False))
    print(f"\nSaved aggregated monthly scores to: {args.output}")


if __name__ == "__main__":
    main()
