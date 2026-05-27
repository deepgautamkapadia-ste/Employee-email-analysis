"""
Task 2: Exploratory Data Analysis for labeled email data.

This script inspects the structure of labeled_data.csv, reports missing values
and dtypes, and creates a small set of visualizations that summarize sentiment
distribution and message activity over time.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from email_analysis_utils import (
    coerce_datetime,
    ensure_visualizations_dir,
    load_csv,
    normalize_month_period,
    pretty_print_section,
    resolve_column_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perform EDA on labeled employee messages.")
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
        "--subject-column",
        default=None,
        help="Subject column name, if not one of the default candidates.",
    )
    parser.add_argument(
        "--body-column",
        default=None,
        help="Body column name, if not one of the default candidates.",
    )
    parser.add_argument(
        "--sentiment-column",
        default=None,
        help="Sentiment label column name, if not one of the default candidates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pretty_print_section("Task 2: Exploratory Data Analysis")

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
    subject_column = resolve_column_name(df, args.subject_column, ["subject", "Subject"], "subject")
    body_column = resolve_column_name(df, args.body_column, ["body", "Body"], "body")
    sentiment_column = resolve_column_name(
        df,
        args.sentiment_column,
        ["Sentiment_Label", "Sentiment", "Label"],
        "sentiment",
    )

    pretty_print_section("Structure")
    print("Shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isna().sum().sort_values(ascending=False))
    print("\nPreview:")
    print(df.head(5).to_string(index=False))
    print("\nText columns in use:")
    print(f"  Subject: {subject_column}")
    print(f"  Body: {body_column}")

    pretty_print_section("Sentiment Summary")
    sentiment_counts = df[sentiment_column].value_counts(dropna=False)
    sentiment_percent = (sentiment_counts / len(df) * 100).round(2)
    summary = pd.DataFrame({"Count": sentiment_counts, "Percent": sentiment_percent})
    print(summary.to_string())

    df[date_column] = coerce_datetime(df[date_column])
    valid_dates = df[date_column].notna().sum()
    print(f"\nValid dates parsed: {valid_dates}/{len(df)}")

    dated_df = df.dropna(subset=[date_column]).copy()
    dated_df["Month"] = normalize_month_period(dated_df[date_column])

    monthly_message_counts = dated_df.groupby("Month").size().sort_index()
    monthly_sentiment_share = (
        dated_df.groupby(["Month", sentiment_column]).size().unstack(fill_value=0).sort_index()
    )
    monthly_sentiment_share = monthly_sentiment_share.div(monthly_sentiment_share.sum(axis=1), axis=0)

    pretty_print_section("Time Trends")
    print("\nMessages per month:")
    print(monthly_message_counts.to_string())
    print("\nSentiment share by month:")
    print((monthly_sentiment_share * 100).round(2).to_string())

    viz_dir = ensure_visualizations_dir()

    # Plot 1: sentiment distribution
    plt.figure(figsize=(8, 5))
    sentiment_counts.reindex(["Positive", "Negative", "Neutral"]).fillna(0).plot(
        kind="bar", color=["#2ca02c", "#d62728", "#7f7f7f"]
    )
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment Label")
    plt.ylabel("Message Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    sentiment_plot_path = viz_dir / "sentiment_distribution.png"
    plt.savefig(sentiment_plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Plot 2: message trend over time
    plt.figure(figsize=(10, 5))
    monthly_message_counts.plot(marker="o", linewidth=2, color="#1f77b4")
    plt.title("Message Volume Over Time")
    plt.xlabel("Month")
    plt.ylabel("Number of Messages")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    trend_plot_path = viz_dir / "message_trends_over_time.png"
    plt.savefig(trend_plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Plot 3: sentiment share over time
    plt.figure(figsize=(10, 5))
    monthly_sentiment_share.reindex(columns=["Positive", "Negative", "Neutral"]).fillna(0).plot(
        kind="line", marker="o", linewidth=2
    )
    plt.title("Sentiment Share Over Time")
    plt.xlabel("Month")
    plt.ylabel("Proportion of Messages")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Sentiment", loc="best")
    plt.tight_layout()
    sentiment_trend_path = viz_dir / "sentiment_trend_over_time.png"
    plt.savefig(sentiment_trend_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("\nSaved visualizations to:", viz_dir)
    print(" -", sentiment_plot_path.name)
    print(" -", trend_plot_path.name)
    print(" -", sentiment_trend_path.name)
    print("\nColumn mapping used:")
    print(f"  Employee: {employee_column}")
    print(f"  Date: {date_column}")
    print(f"  Subject: {subject_column}")
    print(f"  Body: {body_column}")
    print(f"  Sentiment: {sentiment_column}")


if __name__ == "__main__":
    main()
