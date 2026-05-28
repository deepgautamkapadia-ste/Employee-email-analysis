# Employee Email Sentiment Analysis

This project analyzes employee messages stored in `test(in).csv` using a local LLM-based sentiment labeling workflow by default, then performs EDA, scoring, ranking, flight-risk flagging, and a simple predictive model.

The code is split into small, task-focused Python scripts so each step can be run independently and intermediate CSV files can be reused downstream.

## Project Structure

```text
.
├── email_analysis_utils.py
├── task1_sentiment_labeling.py
├── task2_eda.py
├── task3_score_calculation.py
├── task4_employee_ranking.py
├── task5_flight_risk.py
├── task6_predictive_modeling.py
├── test(in).csv
├── labeled_data.csv              # created by Task 1
├── monthly_scores.csv            # created by Task 3
└── visualizations/               # created by Tasks 2, 4, 5, 6
```

## Requirements

Install the following packages:

- `pandas`
- `matplotlib`
- `scikit-learn`
- `transformers`
- `torch`
- `accelerate`
- `openai`

The local model path does not require an OpenAI key.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. If you want to use the OpenAI fallback, set your OpenAI API key:

```bash
setx OPENAI_API_KEY "your_api_key_here"
```

The local default model is:

```bash
Qwen/Qwen2.5-0.5B-Instruct
```

If you want to override the local model, set:

```bash
setx HF_MODEL_NAME "your-local-model-name"
```

Task 1 now checkpoints to `labeled_data.csv` as it runs and can resume from an interrupted run with `--resume`.

## How To Run

Run the scripts in this order:

### 1. Label sentiment

```bash
python task1_sentiment_labeling.py --input "test(in).csv" --output "labeled_data.csv" --provider local --resume --checkpoint-every 50
```

To force the local transformer path explicitly:

```bash
python task1_sentiment_labeling.py --provider local --local-model "Qwen/Qwen2.5-0.5B-Instruct"
```

If you want to use the OpenAI API instead:

```bash
python task1_sentiment_labeling.py --provider openai --openai-model "gpt-4o-mini"
```

For long runs, keep `--resume` enabled. The script rewrites `labeled_data.csv` every `--checkpoint-every` rows so you can rerun it after interruptions without losing progress.

### 2. Run EDA

```bash
python task2_eda.py --input "labeled_data.csv"
```

### 3. Calculate monthly scores

```bash
python task3_score_calculation.py --input "labeled_data.csv" --output "monthly_scores.csv"
```

### 4. Rank employees by month

```bash
python task4_employee_ranking.py --input "monthly_scores.csv"
```

### 5. Flag flight risks

```bash
python task5_flight_risk.py --input "labeled_data.csv"
```

### 6. Train predictive model

```bash
python task6_predictive_modeling.py --labeled-input "labeled_data.csv" --monthly-input "monthly_scores.csv"
```

## Column Mapping

Each script supports argument-based column mapping so you can adapt to different CSV schemas without changing the code.

Typical defaults:

- Employee column: `from`
- Date column: `date`
- Subject column: `subject`
- Body column: `body`
- Sentiment column: `Sentiment_Label`

If your CSV uses different names, pass them explicitly using the script arguments.

## Outputs

- `labeled_data.csv`
  - Original dataset plus `Sentiment_Label`
- `monthly_scores.csv`
  - Employee-month aggregation with monthly sentiment score columns
- `visualizations/sentiment_distribution.png`
- `visualizations/message_trends_over_time.png`
- `visualizations/sentiment_trend_over_time.png`
- `visualizations/employee_rankings_<month>.png`
- `visualizations/flight_risk_summary.png`
- `visualizations/actual_vs_predicted.png`

## Notes On Sentiment Labeling

Task 1 uses an LLM because the messages may include:

- informal or non-work conversation,
- subtle tone,
- sarcasm,
- mixed sentiment,
- short replies that are hard to score with lexicons alone.

Task 1 now defaults to a local Hugging Face causal model and uses the exact chat-template pattern from your snippet. OpenAI is only an optional fallback.

## Results

### Latest Run Snapshot

- Labeled rows: `2,191`
- Sentiment distribution: `1,480 Positive`, `421 Neutral`, `290 Negative`
- Flight-risk employees: none met the 4-negative-messages-in-30-days threshold
- Predictive model: `MSE = 1.4636`, `MAE = 0.9928`, `R-squared = 0.4674`

### Top 3 Positive Employees Overall

| Employee | Total Monthly Score |
|---|---:|
| `sally.beck@enron.com` | `63` |
| `lydia.delgado@enron.com` | `60` |
| `johnny.palmer@enron.com` | `50` |

### Top 3 Negative Employees Overall

| Employee | Total Monthly Score |
|---|---:|
| `rhonda.denton@enron.com` | `33` |
| `don.baughman@enron.com` | `36` |
| `john.arnold@enron.com` | `40` |

### Flagged Flight Risks

- None in the latest run

## Key Insights

- The latest run produced a strong positive skew overall, with far more positive than negative labels.
- Negative sentiment was concentrated enough to be useful for monthly scoring, but not enough to trigger the flight-risk rule.
- The month-by-month ranking and plots are the main artifacts to review for operational follow-up.

## Recommendations

- Review the months with the lowest monthly scores and compare them with team events or organizational changes.
- Use the ranking plots to identify consistently positive contributors and consistently strained months.
- If you need more sensitive flight-risk detection, lower the threshold or widen the rolling window logic.

## Optional Improvements

- Save API call failures to a retry log.
- Add caching to disk for repeated message texts.
- Add more features for predictive modeling, such as response time or communication network metrics.
- Replace the LLM sentiment step with a local transformer pipeline when API access is not desired.
- Add a progress log so long Task 1 runs can be monitored outside the console.
