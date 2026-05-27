"""
Task 6: Predictive modeling.

This script engineers monthly features from labeled_data.csv, joins them to the
monthly sentiment scores, and trains a Linear Regression model to predict the
monthly score.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from email_analysis_utils import (
    coerce_datetime,
    combine_text_columns,
    ensure_visualizations_dir,
    load_csv,
    normalize_month_period,
    pretty_print_section,
    resolve_column_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model to predict monthly sentiment score.")
    parser.add_argument("--labeled-input", default="labeled_data.csv", help="Input labeled CSV path.")
    parser.add_argument("--monthly-input", default="monthly_scores.csv", help="Input monthly scores CSV path.")
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
        "--monthly-month-column",
        default=None,
        help="Month column in monthly_scores.csv, if not one of the default candidates.",
    )
    parser.add_argument(
        "--target-column",
        default=None,
        help="Target score column in monthly_scores.csv, if not one of the default candidates.",
    )
    return parser.parse_args()


def build_feature_table(
    labeled_df: pd.DataFrame,
    employee_column: str,
    date_column: str,
    subject_column: str,
    body_column: str,
    sentiment_column: str,
) -> pd.DataFrame:
    """Aggregate message-level text features into employee-month rows."""

    labeled_df = labeled_df.copy()
    labeled_df[date_column] = coerce_datetime(labeled_df[date_column])
    labeled_df = labeled_df.dropna(subset=[date_column]).copy()
    labeled_df["Month_Year"] = normalize_month_period(labeled_df[date_column])
    labeled_df["Email_Text"] = combine_text_columns(labeled_df[subject_column], labeled_df[body_column])

    labeled_df["Message_Length_Chars"] = labeled_df["Email_Text"].str.len()
    labeled_df["Word_Count"] = labeled_df["Email_Text"].str.split().apply(len)
    labeled_df["Sentence_Count"] = labeled_df["Email_Text"].str.count(r"[.!?]+").clip(lower=0) + 1
    labeled_df["Has_Negative"] = labeled_df[sentiment_column].eq("Negative")

    features = (
        labeled_df.groupby([employee_column, "Month_Year"], as_index=False)
        .agg(
            Monthly_Message_Frequency=("Email_Text", "size"),
            Avg_Message_Length_Chars=("Message_Length_Chars", "mean"),
            Avg_Word_Count=("Word_Count", "mean"),
            Avg_Sentence_Count=("Sentence_Count", "mean"),
            Max_Message_Length_Chars=("Message_Length_Chars", "max"),
            Negative_Message_Ratio=("Has_Negative", "mean"),
        )
    )

    return features


def main() -> None:
    args = parse_args()
    pretty_print_section("Task 6: Predictive Modeling")

    labeled_df = load_csv(args.labeled_input)
    monthly_df = load_csv(args.monthly_input)

    employee_column = resolve_column_name(
        labeled_df,
        args.employee_column,
        ["from", "From", "employee", "Employee", "Employee_ID", "EmployeeID", "Employee Name", "Name"],
        "employee",
    )
    date_column = resolve_column_name(
        labeled_df,
        args.date_column,
        ["date", "Date", "Timestamp", "Datetime", "Time", "Created_At"],
        "date",
    )
    subject_column = resolve_column_name(labeled_df, None, ["subject", "Subject"], "subject")
    body_column = resolve_column_name(labeled_df, None, ["body", "Body"], "body")
    sentiment_column = resolve_column_name(
        labeled_df,
        None,
        ["Sentiment_Label", "Sentiment", "Label"],
        "sentiment",
    )

    monthly_month_column = resolve_column_name(
        monthly_df,
        args.monthly_month_column,
        ["Month_Year", "Month", "MonthYear", "Period"],
        "month",
    )
    target_column = resolve_column_name(
        monthly_df,
        args.target_column,
        ["Monthly_Score", "Cumulative_Monthly_Score", "Score"],
        "target score",
    )

    features = build_feature_table(
        labeled_df,
        employee_column,
        date_column,
        subject_column,
        body_column,
        sentiment_column,
    )

    train_df = monthly_df[[employee_column, monthly_month_column, target_column]].copy()
    merged = train_df.merge(
        features,
        left_on=[employee_column, monthly_month_column],
        right_on=[employee_column, "Month_Year"],
        how="inner",
    )

    if merged.shape[0] < 2:
        raise ValueError("Not enough monthly rows after feature join to train a model.")

    feature_columns = [
        "Monthly_Message_Frequency",
        "Avg_Message_Length_Chars",
        "Avg_Word_Count",
        "Avg_Sentence_Count",
        "Max_Message_Length_Chars",
        "Negative_Message_Ratio",
    ]
    X = merged[feature_columns].fillna(0)
    y = merged[target_column].fillna(0)

    test_size = 0.2 if len(merged) >= 5 else 0.5
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    pretty_print_section("Model Evaluation")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R-squared: {r2:.4f}")

    viz_dir = ensure_visualizations_dir()
    plot_path = viz_dir / "actual_vs_predicted.png"

    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, predictions, alpha=0.8, color="#1f77b4")
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="#d62728", linewidth=2)
    plt.title("Actual vs Predicted Monthly Sentiment Score")
    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nSaved actual-vs-predicted plot to: {plot_path}")


if __name__ == "__main__":
    main()
