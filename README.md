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

## How To Run

Run the scripts in this order:

### 1. Label sentiment

```bash
python task1_sentiment_labeling.py --input "test(in).csv" --output "labeled_data.csv"
```

To force the local transformer path explicitly:

```bash
python task1_sentiment_labeling.py --provider local --local-model "Qwen/Qwen2.5-0.5B-Instruct"
```

If you want to use the OpenAI API instead:

```bash
python task1_sentiment_labeling.py --provider openai --openai-model "gpt-4o-mini"
```

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

### Top 3 Positive Employees

| Month | Employee | Score |
|---|---:|---:|
| [FILL IN] | [FILL IN] | [FILL IN] |
| [FILL IN] | [FILL IN] | [FILL IN] |
| [FILL IN] | [FILL IN] | [FILL IN] |

### Top 3 Negative Employees

| Month | Employee | Score |
|---|---:|---:|
| [FILL IN] | [FILL IN] | [FILL IN] |
| [FILL IN] | [FILL IN] | [FILL IN] |
| [FILL IN] | [FILL IN] | [FILL IN] |

### Flagged Flight Risks

- [FILL IN]

## Key Insights

- [FILL IN: e.g., which months had the highest volume of negative messages]
- [FILL IN: e.g., whether sentiment trends changed over time]
- [FILL IN: e.g., which employees or teams need attention]

## Recommendations

- [FILL IN: e.g., coach flagged employees with repeated negative bursts]
- [FILL IN: e.g., review high-volume negative months for root causes]
- [FILL IN: e.g., compare monthly score trends with team events or policy changes]

## Optional Improvements

- Save API call failures to a retry log.
- Add caching to disk for repeated message texts.
- Add more features for predictive modeling, such as response time or communication network metrics.
- Replace the LLM sentiment step with a local transformer pipeline when API access is not desired.
