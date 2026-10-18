# hm-recsys

Product recommendations for a fashion retailer on the public
[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
dataset (real transactions), plus an experimentation module on how to measure a recommender change
online (A/B test analysis, power, variance reduction).

Status: data loading, splitting and evaluation metrics are in place. No model results yet; this
README only reports numbers that were actually measured.

## Setup

```bash
pip install -e ".[dev]"
pytest -q            # runs on synthetic fixtures, no network or data needed
ruff check . && ruff format --check .
```

## Data

Raw data is not committed (Kaggle competition licence and size). To get it:

1. Accept the competition rules on Kaggle and configure the Kaggle CLI token
   (`~/.kaggle/kaggle.json`, never committed).
2. `python scripts/download_data.py` fetches `transactions_train.csv`, `articles.csv`, `customers.csv`
   into `data/raw/`.
3. `python scripts/prepare_data.py --weeks 16 --customer-fraction 0.1` writes the subsample to
   `data/processed/transactions.parquet` and the sample description to `data/processed/sample.json`.
4. `python scripts/eda.py` computes the exploratory numbers into `results/eda.json`.

Every result reported below states the sample it was computed on.

## Evaluation protocol

- Time-based split only. Train = everything before the validation week, validation = the next
  7 days, test = the last 7 days (touched once at the end).
- Task: for each customer who bought something in the target week, predict 12 articles.
- Metrics: MAP@12 (competition metric), recall@12, NDCG@12; recall@100/200 for candidate generation.
- Customers with no purchases in the target week are not evaluated.

## Design decisions

| Decision | Choice | Why |
|---|---|---|
| Split | Last 7 days = test, 7 days before = validation, rest = train | Mirrors how the model would be used: predict the next week from the past. Random splits would leak future purchases. |
| Leakage guard | `history_before(df, cutoff)` is the only way to get rows for features; a test checks that no validation or test row is in it | Features for a prediction week may only use data from before that week. |
| Subsample | Last N weeks, optional customer sample chosen by `hash(customer_id)` | Runs on a laptop. Hashing keeps the same customers in every run and every week, so a customer is never half in the sample. |
| AP@12 denominator | `min(#relevant, 12)` | Same as the competition, so numbers are comparable to the leaderboard definition. |
| Missing prediction | Scores 0 | A customer we cannot serve is a failure, not something to skip. |
| Storage | DuckDB reads the 31M-row CSV, writes parquet | Handles the CSV without loading it into pandas. |

## Exploratory analysis

To be filled with the output of `scripts/eda.py` once the data is downloaded.

## Layout

```
src/hm_recsys/metrics.py   MAP@k, recall@k, NDCG@k
src/hm_recsys/data.py      DuckDB loading, subsampling, time split, leakage helper
scripts/                   download, prepare, EDA
tests/                     unit tests on synthetic fixtures
```
