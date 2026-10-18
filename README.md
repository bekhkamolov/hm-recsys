# hm-recsys

Product recommendations for a fashion retailer on the public
[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
dataset (real transactions), plus an experimentation module on how to measure a recommender change
online (A/B test analysis, power, variance reduction).

Status: data loading, splitting, evaluation metrics and EDA are done (Day 1). No model results yet;
this README only reports numbers that were actually measured.

## Setup

```bash
pip install -e ".[dev]"
pytest -q            # runs on synthetic fixtures, no network or data needed
ruff check . && ruff format --check .
```

## Data

Raw data is not committed (Kaggle competition licence and size). To get it:

1. Accept the competition rules on Kaggle and configure the Kaggle CLI token
   (`~/.kaggle/access_token` or `~/.kaggle/kaggle.json`, never committed).
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

Computed by `scripts/eda.py` on the **last 16 weeks (2020-06-03 to 2020-09-22), 10% hashed customer
sample: 514,885 purchase rows, 59,640 customers, 29,537 distinct articles.** Split on this sample:
validation 2020-09-09 to 2020-09-15, test 2020-09-16 to 2020-09-22.

| Quantity | Value |
|---|---|
| Purchases per customer (mean) | 8.6 |
| Purchases per customer (25 / 50 / 75 / 90th percentile) | 3 / 5 / 11 / 19 |
| Repeat-purchase rate (share of purchases where the customer bought that article earlier in the window) | 12.0% |
| Share of purchases going to the top 1% of articles | 13.9% |
| Share of purchases going to the top 10% of articles | 54.9% |

What this means for the models:
- **Heavy-tailed activity.** The median customer made 5 purchases in 16 weeks and the 90th percentile 19,
  so many customers have little history. Segment results (new vs. returning) will matter.
- **Repeat purchases are a real signal, but a minority.** 12% of purchases repeat an article already
  bought in the window. This is a lower bound, because the window is only 16 weeks. "Buy again" is
  therefore worth testing as a baseline, but it cannot cover most purchases.
- **Strong popularity skew.** 10% of articles carry 55% of purchases, so popularity baselines will be
  hard to beat. Every later model is compared against them.
- **Seasonality / non-stationarity.** Weekly purchases are not flat: 22.9k in the first (partial) week, a
  peak of 49.0k and 56.4k in the weeks of 15 and 22 June, then about 25-31k per week. The June peak
  is probably a sale period, but that has not been verified in the data. It means "popular last week"
  changes over time, which is why popularity is computed on the most recent week only. The first
  and last weeks in the table are partial (the window starts on a Wednesday and ends on a
  Tuesday), so their counts are lower for that reason.

## Layout

```
src/hm_recsys/metrics.py   MAP@k, recall@k, NDCG@k
src/hm_recsys/data.py      DuckDB loading, subsampling, time split, leakage helper
scripts/                   download, prepare, EDA
tests/                     unit tests on synthetic fixtures
```
