"""
Task 7: Negative-trend flight risk summary.

This small helper script identifies the employees with the highest total number
of Negative messages in the labeled dataset. The top 5 are marked as a simple
"trend-based flight risk" list for quick review.

Output:
    - Prints the top 5 negative-message employees to the terminal.
    - Saves a CSV summary to negative_trend_flight_risk.csv.
    - Saves a bar chart to visualizations/negative_trend_flight_risk.png.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from email_analysis_utils import ensure_visualizations_dir, load_csv, pretty_print_section, resolve_column_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank employees by total negative message count.")
    parser.add_argument("--input", default="labeled_data.csv", help="Input labeled CSV path.")
    parser.add_argument("--output", default="negative_trend_flight_risk.csv", help="Output CSV path.")
    parser.add_argument(
        "--employee-column",
        default=None,
        help="Employee identifier column name, if not one of the default candidates.",
    )
    parser.add_argument(
        "--sentiment-column",
        default=None,
        help="Sentiment label column name, if not one of the default candidates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pretty_print_section("Task 7: Negative Trend Flight Risk")

    df = load_csv(args.input)
    employee_column = resolve_column_name(
        df,
        args.employee_column,
        ["from", "From", "employee", "Employee", "Employee_ID", "EmployeeID", "Employee Name", "Name"],
        "employee",
    )
    sentiment_column = resolve_column_name(
        df,
        args.sentiment_column,
        ["Sentiment_Label", "Sentiment", "Label"],
        "sentiment",
    )

    negative_counts = (
        df.loc[df[sentiment_column].eq("Negative")]
        .groupby(employee_column)
        .size()
        .reset_index(name="Negative_Message_Count")
        .sort_values(["Negative_Message_Count", employee_column], ascending=[False, True])
        .reset_index(drop=True)
    )

    top_5 = negative_counts.head(5).copy()
    top_5["Negative_Trend_Flight_Risk"] = True
    top_5["Rank"] = range(1, len(top_5) + 1)

    pretty_print_section("Top 5 Negative Employees")
    if top_5.empty:
        print("No negative messages found in the dataset.")
    else:
        print(top_5[[ "Rank", employee_column, "Negative_Message_Count", "Negative_Trend_Flight_Risk" ]].to_string(index=False))

    top_5.to_csv(args.output, index=False)
    print(f"\nSaved summary CSV to: {args.output}")

    viz_dir = ensure_visualizations_dir()
    plot_path = viz_dir / "negative_trend_flight_risk.png"

    plt.figure(figsize=(10, 5))
    if top_5.empty:
        plt.text(0.5, 0.5, "No negative messages found", ha="center", va="center", fontsize=14)
        plt.axis("off")
    else:
        plt.barh(
            top_5[employee_column].astype(str),
            top_5["Negative_Message_Count"],
            color="#d62728",
        )
        plt.gca().invert_yaxis()
        plt.title("Top 5 Employees by Total Negative Messages")
        plt.xlabel("Negative Message Count")
        plt.ylabel("Employee")
        plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization to: {plot_path}")


if __name__ == "__main__":
    main()

