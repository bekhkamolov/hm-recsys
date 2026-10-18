# Data, baselines and candidate generation

Part of [hm-recsys](../README.md). Every number here comes from the JSON files in `results/`.

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
6. `python scripts/candidates.py` measures candidate recall per source and for the union.
7. `python scripts/ranker.py --train-weeks 4` trains the ranker and writes the validation results,
   ablation and training-weeks comparison (about 15 minutes and 6+ GB of free memory).
8. `python scripts/download_ab_data.py` and `python scripts/ab_cookie_cats.py` run the A/B test analysis
   on the public Cookie Cats data (needs only the Kaggle token, no H&M data).
9. `python scripts/experiment_sim.py` runs the power, CUPED and peeking simulations on H&M behaviour.
10. `python scripts/drift_check.py` runs the drift and model-staleness experiment (roughly 10 to 15 minutes);
    `python scripts/smoke_test.py <url>` checks a running service.
11. `python scripts/build_serving.py` builds the serving artifacts; `python scripts/track_results.py`
    logs the saved results to MLflow (see [serving and monitoring](serving-and-monitoring.md)).
12. `python scripts/final_eval.py --target val` is a dry run of the final evaluation on the validation week;
    `python scripts/final_eval.py --target test` is the single evaluation of the held-out test week. Run the test
    one only once you have finished choosing models and settings.

Each script writes a JSON file to `results/`, which is what the tables in these documents are taken from. The expensive
steps (the ranker, drift and final evaluation scripts) take roughly 5 to 15 minutes each on a laptop and need several GB of
free memory.

Dependencies are split into extras: the base install is what the serving image needs; `.[train]` adds
DuckDB, SciPy, `implicit` and scikit-learn; `.[serve]` adds FastAPI and uvicorn; `.[tracking]` adds
MLflow; `.[dev]` installs all of them plus pytest and ruff.

Every result reported below states the sample it was computed on.

## Evaluation protocol

- Time-based split only. Train = everything before the validation week, validation = the next
  7 days, test = the last 7 days (touched once at the end).
- Task: for each customer who bought something in the target week, predict 12 articles.
- Metrics: MAP@12 (competition metric), recall@12, NDCG@12; recall@100/200 for candidate generation.
- Customers with no purchases in the target week are not evaluated.

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

## Candidate generation (Day 3)

The ranker will only re-order what the candidate sources propose, so here the question is recall: does
the article a customer buys in the validation week appear in their list at all? Same validation
week and sample as above (**7,256 customers, 10% customer sample, purchases before 2020-09-09**).
Each source proposes up to 100 articles per customer. Produced by `python scripts/candidates.py`;
raw numbers in `results/candidates_val.json`. "Recall" is the mean over customers of the share of
their purchased articles that are in the list; "customers with a hit" is the share of customers
for whom at least one purchased article is in the list.

| Source | recall@100 | Customers with a hit | Customers served | Avg list size |
|---|---|---|---|---|
| Buy again (own past purchases) | 0.0378 | 7.4% | 76.1% | 9.4 |
| Item-to-item co-purchase | 0.0158 | 3.8% | 76.0% | 73.3 |
| ALS (`implicit`) | 0.0380 | 9.3% | 76.1% | 76.1 |
| Popularity (last 7 days, top 100) | 0.0941 | 22.2% | 100% | 100 |

Union of all four (buy again, co-purchase, ALS, popularity, duplicates removed):

| Candidate set | Recall | Customers with a hit | Avg list size |
|---|---|---|---|
| All four sources, no cap | **0.1611** | 33.1% | 249 |
| Same, first 200 in priority order | 0.1088 | 22.7% | 176 |
| Without buy again | 0.1287 | 28.8% | 240 |
| Without co-purchase | 0.1531 | 31.9% | 178 |
| Without ALS | 0.1396 | 29.6% | 182 |
| Without popularity | 0.0845 | 16.9% | 156 |

By customer segment: returning customers (5,523) get union recall 0.1813 with about 296 candidates;
customers with no history (1,733) get popularity only, recall 0.0968 with 100 candidates.

What the numbers say:
- **Popularity is the strongest single source at 100 candidates (0.094), more than twice ALS or buy
  again.** Recent popularity is hard to beat, as in Day 2. It is also the only source that serves
  new customers (24% of the evaluation population).
- **Every source adds something to the union, but unevenly.** Removing popularity costs 0.077
  recall, buy again 0.032, ALS 0.022 and co-purchase only 0.008. Co-purchase is the weakest and the
  most redundant; it is kept for now because it still adds, but it is the first candidate to drop if
  the ranker gets too slow.
- **ALS is modest.** Its recall@100 equals buy again's, but it reaches more customers (9.3% vs 7.4%
  with a hit) and it only proposes articles the customer has not bought. As a sanity check that
  it is not broken or badly configured, I tried 32 / 64 / 128 factors on the validation week: recall@100
  was 0.034 / 0.038 / 0.037 with already-bought articles filtered and 0.045 / 0.050 / 0.053 without the
  filter. The number of factors barely matters. Allowing bought articles raises ALS recall, but
  owned articles are already proposed by the buy-again source, so the filter stays (I did not measure the overlap). These variants were looked
  at on the validation week; the setting was not changed because of them.
- **The union reaches 16.1% recall with 249 candidates on average**, against 4.6% recall@12 for the
  best Day 2 baseline. These are different measures (recall over ~250 candidates vs. over 12
  recommendations), so this is headroom for the ranker, not a result: the ranker still has to pick
  the right 12.
- **Ceiling.** 89.3% of the validation-week purchases are of articles that were sold before the
  cutoff in the window, so sources built from history cannot exceed that. Union recall of
  16.1% is far below that, so most of the gap is purchases the sources do not propose, not
  articles that were never sold.
- **Cutting the union to 200 loses a lot (0.1611 to 0.1088)**, because popularity comes last in my
  priority order and gets cut. The ranker should see the full union, or the merge needs a smarter
  order. This has not been optimised.
