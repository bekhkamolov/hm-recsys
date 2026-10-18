# hm-recsys

Product recommendations for a fashion retailer on the public
[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
dataset (real transactions), plus an experimentation module on how to measure a recommender change
online (A/B test analysis, power, variance reduction).

Status: data, evaluation and baselines are done (Days 1-2). No learned models yet; this README only
reports numbers that were actually measured.

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
   It also writes `data/processed/customers.parquet` (customer id and age) for the sampled customers.
4. `python scripts/eda.py` computes the exploratory numbers into `results/eda.json`.
5. `python scripts/baselines.py` evaluates the baselines on the validation week.

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
| Popularity window | Last 7 days before the cutoff | Purchases are strongly non-stationary (see weekly counts), so recent popularity is a better guess than all-time popularity. Not tuned. |
| Popularity ties | Broken by lower article id | Deterministic results. |
| Age groups | <25, 25-34, 35-44, 45-54, 55+, unknown | Coarse buckets so each has enough customers in a 10% sample. A group with fewer than 12 articles is padded from global popularity. |
| Buy again order | Most recent distinct purchase first | Simplest recency rule. Frequency-based ordering was not tried. |
| Fallback for buy again | Fill remaining slots from popularity, never repeat an item | Customers with fewer than 12 distinct past items still get 12 recommendations. |
| Segments | Returning vs. no history (any purchase before the target week in the loaded window), and age group | Returning/no-history is a proxy: it cannot see purchases older than the window. |
| Reference baseline | Buy again + global popularity | The age-group variant did not beat it, so the simpler one is kept (see Baselines). |

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

## Baselines (Day 2)

Validation week 2020-09-09 to 2020-09-15, same 10% customer sample as above, **7,256 customers**
(everyone in the sample who bought something that week). All models predict 12 articles from
purchases before 2020-09-09 only. Produced by `python scripts/baselines.py`; raw numbers in
`results/baselines_val.json`. The test week is not used.

| Model | MAP@12 | recall@12 | NDCG@12 |
|---|---|---|---|
| Global popularity (last 7 days) | 0.0053 | 0.0198 | 0.0113 |
| Popularity by age group (last 7 days) | 0.0055 | 0.0175 | 0.0106 |
| Buy again (own purchases, then global popularity) | 0.0233 | 0.0458 | 0.0339 |
| Buy again + age-group popularity | 0.0235 | 0.0447 | 0.0336 |
| Buy again, max 6 own items + age-group popularity | 0.0233 | 0.0436 | 0.0332 |

MAP@12 by segment (same models, number of customers in brackets):

| Model | Returning (5,523) | No history (1,733) |
|---|---|---|
| Global popularity | 0.0051 | 0.0062 |
| Popularity by age group | 0.0050 | 0.0068 |
| Buy again + global popularity | 0.0287 | 0.0062 |
| Buy again + age-group popularity | 0.0288 | 0.0068 |

Age-group MAP@12 for each model is in `results/baselines_val.json`; the groups have between 750 and
2,269 customers (29 with unknown age), so differences between them are noisy.

What the numbers say:
- **"Buy again" is the first real step.** It lifts MAP@12 from 0.0053 to 0.0233, about 4.4x, and
  recall@12 from 0.020 to 0.046. The whole gain is on returning customers (0.0051 to 0.0287); for customers
  without history it falls back to popularity and is identical to it.
- **Age-group popularity did not beat global popularity.** MAP@12 is 0.0055 vs 0.0053, which is a tie
  at this sample size, while recall@12 and NDCG@12 are lower. It helps the under-25s (0.0084
  vs 0.0069) and the 55+ (0.0064 vs 0.0043) and hurts the 25-54 groups, so the overall tie hides
  opposite effects in small groups. No confidence interval has been computed, so treat any
  MAP difference below about 0.001 as noise.
- **Because of that, the reference baseline is "buy again + global popularity".** Replacing the
  fallback with age-group popularity gives 0.0235 vs 0.0233 MAP@12 but lower recall@12 and NDCG@12,
  so it is not an improvement and the simpler version is kept as the bar the ranker must beat.
- **Capping own items at 6 made things worse** (0.0233 vs 0.0235 MAP, recall@12 0.0436 vs 0.0447), so
  the uncapped version is used. Only the value 6 was tried, so this does not rule out other splits.
- **Absolute numbers are low and not comparable to the Kaggle leaderboard.** This is a 10% customer
  sample, one validation week, and the population is only customers who bought that week. The
  comparison between rows is what matters.
- **No-history customers are 24% of the evaluation population** (1,733 of 7,256) and get only
  popularity. "No history" means no purchase in the 16-week window; some may be older H&M customers.
  This segment is where candidate sources beyond the customer's own purchases (Day 3) should help.

## Layout

```
src/hm_recsys/metrics.py   MAP@k, recall@k, NDCG@k, per-segment evaluation
src/hm_recsys/data.py      DuckDB loading, subsampling, time split, leakage helper
src/hm_recsys/baselines.py popularity, age-group popularity, buy again, fallback filling
scripts/                   download, prepare, EDA, baselines
tests/                     unit tests on synthetic fixtures
```
