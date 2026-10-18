# hm-recsys

Product recommendations for a fashion retailer on the public
[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
dataset (real transactions), plus an experimentation module on how to measure a recommender change
online (A/B test analysis, power, variance reduction).

Status: data, evaluation, baselines, candidate generation and the ranker are done (Days 1-5), as are the real A/B test analysis (Day 6) and the experiment design simulations (Day 7); this README only
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
6. `python scripts/candidates.py` measures candidate recall per source and for the union.
7. `python scripts/ranker.py --train-weeks 4` trains the ranker and writes the validation results,
   ablation and training-weeks comparison (about 15 minutes and 6+ GB of free memory).
8. `python scripts/download_ab_data.py` and `python scripts/ab_cookie_cats.py` run the A/B test analysis
   on the public Cookie Cats data (needs only the Kaggle token, no H&M data).
9. `python scripts/experiment_sim.py` runs the power, CUPED and peeking simulations on H&M behaviour.

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
| Candidates per source | 100 | Large enough for recall@100 to be meaningful, small enough for a ranker to score. Not tuned. |
| Co-purchase | Cosine similarity of articles over customers who bought both (whole window, not same basket); 50 nearest neighbours per article; seeds = customer's 10 latest purchases; already-bought articles excluded | The data has no baskets beyond same-day purchases, which are sparse in a 10% sample. Neighbour pruning keeps the matrix small. |
| ALS | `implicit`, 64 factors, 15 iterations, regularisation 0.01, confidence 1 + log(1 + purchases), already-bought articles excluded, seed 0 | Defaults with a mild repeat-purchase weight. Buy again already covers owned articles, so ALS should add new ones. Factors 32/128 were checked on validation and made little difference. |
| Union order | buy again, co-purchase, ALS, popularity; duplicates removed | Own history first. Order only matters if the union is cut. |
| Candidate metric | Mean share of purchased articles found anywhere in the list, plus share of customers with at least one hit | The ranker sees the whole list, so what counts is whether the article is in it, not its position. |
| Ranker training set | Candidates and features rebuilt as of the start of each of 4 weeks before validation, labelled with that week's purchases; every candidate not bought is a negative; customers with no purchased candidate dropped from training | The same procedure used at prediction time, so train and validation rows look alike, and no row sees its own label week. The same customer in two weeks is two ranking groups. |
| Training weeks | 4 | More weeks helped up to 4 (1, 2, 4, 6 compared, 3 seeds each) and 6 was no clearer. Chosen on the validation week, so validation is slightly optimistic. |
| Ranker settings | LambdaRank, 300 trees, lr 0.05, 31 leaves, min 50 rows per leaf, 80% subsampling, no early stopping | Fixed and untuned, so the validation week is not used to fit the model. |
| Feature encoding | Category ids (department, colour, product type, ...) as dense codes treated as categorical by LightGBM; features stored as float32 | Raw ids are arbitrary labels and some are huge. float32 halves memory, which mattered on a 16 GB laptop. |
| Collaborative scores | ALS dot product and co-purchase score computed for every candidate, not only those from that source | A candidate from another source still gets the information. |
| A/B dataset | Cookie Cats (gate 30 vs 40) | Real users, user-level assignment, two binary metrics and one heavy-tailed metric. Limits are in its section. |
| A/B tests | Two-sided; pooled z for the p-value, unpooled SE for the CI; bootstrap as a cross-check | Standard for two independent proportions; the bootstrap does not depend on the normal approximation. |
| Practical threshold | 0.5 pp of retention (assumption) | Needed to separate "significant" from "worth acting on"; should come from business value, not from the data. |
| Experiment windows | 56-day pre-period, 14-day outcome from 2020-08-12; population = customers with a purchase in the pre-period | Before the validation and test weeks. A 14-day window covers two weekly cycles. Customers with no recent purchase are not in the data as non-buyers. |
| Treatment effect in simulations | Multiply each treated customer's outcome by (1 + uplift) | Simple and keeps the link between past and future spend that CUPED relies on. One possible model, not the only one. |
| CUPED covariate | Pre-period spend, theta estimated on both arms pooled | Known before assignment, so it cannot be affected by the treatment. A second covariate (order count) added nothing. |
| Sequential testing | 14 daily looks; Pocock and O'Brien-Fleming-style constants calibrated by simulating the null | The exact boundaries need no tables and match the textbook value in a test. |
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

## Ranker (Days 4-5)

LightGBM LambdaRank (`LGBMRanker`) re-orders the union of the four candidate sources (about 250
candidates per customer) and the top 12 are the recommendation. 33 features in six groups
(customer, article, customer x article history, age-group popularity, ALS / co-purchase scores,
source ranks), all computed from purchases before the prediction week.

**Training data.** For each of the 4 weeks before validation, candidates and features are built
as of that week's start (from earlier purchases only) and labelled with that week's purchases:
7.34M candidate rows, 14,040 purchased (0.19%). Validation: candidates built as of 2020-09-09,
1.81M rows for the 7,256 customers, of which 3,391 are purchased (the candidates contain 14.6% of
the 23,205 purchases that week). Fixed settings, no early stopping on validation: 300 trees, learning
rate 0.05, 31 leaves, min 50 rows per leaf, 80% row and feature subsampling. None of these were
tuned. Same 10% customer sample and validation week as above. Produced by
`python scripts/ranker.py --train-weeks 4`; raw numbers in `results/ranker_val.json`.

| Model | MAP@12 | recall@12 | NDCG@12 |
|---|---|---|---|
| Buy again + global popularity (Day 2 reference) | 0.0233 | 0.0458 | 0.0339 |
| **LightGBM ranker, all features** (3 seeds) | **0.0279 / 0.0281 / 0.0277** | 0.0556 / 0.0539 / 0.0549 | 0.0407 / 0.0402 / 0.0403 |

The ranker beats the reference on all three metrics, by about 20% relative on MAP@12 (0.0279 vs
0.0233) and on recall@12. A paired bootstrap over the 7,256 customers (2,000 resamples, seed-0
model) gives a MAP@12 difference of +0.0045 with a 95% interval of [0.0028, 0.0062]. Two caveats:
that interval only covers which customers happened to buy that week, not other samples or other
training seeds, and the number of training weeks was chosen on this same validation week (below),
so 0.0279 is a little optimistic. The test week has not been used.

**Does more training data help?** Yes, up to a point (full model, 3 seeds each):

| Training weeks | Candidate rows | Purchased rows | MAP@12 (mean of 3 seeds) | recall@12 (mean) |
|---|---|---|---|---|
| 1 | 1.88M | 3,675 | 0.0246 | 0.0493 |
| 2 | 3.86M | 7,499 | 0.0257 | 0.0519 |
| **4** | 7.34M | 14,040 | 0.0279 | 0.0548 |
| 6 | 10.92M | 21,039 | 0.0281 | 0.0566 |

The step from 2 to 4 weeks (+0.0022) is more than five times the spread between seeds (0.0004). From 4 to 6 weeks the
mean barely changes and the seeds spread by 0.002 (0.0273 to 0.0293), so 6 is not
distinguishably better and 4 was kept. With one week of training the ranker is only 0.0013 above the
reference, so the gain depends on enough training data; a 10% customer sample is small.

**Feature-group ablation** (one seed, seed 0, whose full-model score is 0.0279; the three full-model
seeds differ by about +-0.0002, so differences below about 0.0005 are not reliable):

| Features | MAP@12 | recall@12 | NDCG@12 |
|---|---|---|---|
| All features | 0.0279 | 0.0556 | 0.0407 |
| Without customer | 0.0265 | 0.0541 | 0.0391 |
| Without article | 0.0268 | 0.0535 | 0.0390 |
| Without customer x article history | 0.0258 | 0.0530 | 0.0382 |
| Without age-group popularity | 0.0274 | 0.0542 | 0.0398 |
| Without ALS / co-purchase scores | 0.0273 | 0.0539 | 0.0395 |
| Without source ranks | 0.0277 | 0.0554 | 0.0405 |
| Only source ranks | 0.0239 | 0.0484 | 0.0349 |
| Only customer x article history | 0.0229 | 0.0407 | 0.0318 |
| Only article | 0.0057 | 0.0228 | 0.0120 |
| Only ALS / co-purchase scores | 0.0053 | 0.0167 | 0.0101 |
| Only age-group popularity | 0.0049 | 0.0158 | 0.0097 |
| Only customer | 0.0028 | 0.0109 | 0.0060 |

- **Customer x article history matters most**: removing it costs 0.0021 MAP@12. This group has
  repurchase count, days since the customer last bought the article, price vs. the customer's usual
  price, and the share of their past purchases in the same product type and colour.
- **Customer and article features each add about 0.001** (0.0014 and 0.0011).
- **ALS / co-purchase scores and age-group popularity add little** (0.0006 and 0.0005, near the noise
  level), and **source ranks add nothing measurable** (0.0002). The model already has the same
  information from other features; the ranks are redundant.
- **No single group beats the reference alone** (best: source ranks 0.0239 vs. 0.0233, within noise). The
  gain comes from combining groups, and the single-group runs of customer, article, ALS and
  age-popularity features are far below the reference: they cannot rank without knowing what the
  customer already bought.

**Feature importance** (share of split gain, full model, seed 0). Top features: article department
23.6%, days since the customer last bought the article 14.1%, rank in the buy-again list 5.3%, product
type 5.1%, colour 4.8%, days since the customer's last purchase 4.8%, article popularity over 28 days
3.9%, co-purchase score 3.6%. Read this with care. `a_department` is a categorical feature with
many values, which tends to collect a lot of gain, and removing the whole article group costs only
0.0011 MAP@12. I did not test removing `a_department` alone. The ablation is the better guide to
what matters.

**By segment** (MAP@12, ranker seed 0 vs. reference):

| Segment | Customers | Ranker | Reference |
|---|---|---|---|
| Returning | 5,523 | 0.0343 | 0.0287 |
| No history | 1,733 | 0.0075 | 0.0062 |
| Age < 25 | 1,938 | 0.0265 | 0.0188 |
| Age 25-34 | 2,269 | 0.0255 | 0.0232 |
| Age 35-44 | 750 | 0.0345 | 0.0268 |
| Age 45-54 | 1,344 | 0.0292 | 0.0275 |
| Age 55+ | 926 | 0.0295 | 0.0246 |
| Age unknown | 29 | 0.0147 | 0.0134 |

The ranker is ahead in every segment on MAP@12, but only the returning customers are a clear win:
for customers with no history it is mixed (MAP@12 0.0075 vs 0.0062, but recall@12 0.0216 vs 0.0234)
and no significance test was run per segment. The age groups are small, so those differences are noisy.

**Limits.** The absolute numbers are low (MAP@12 about 0.03): most purchases, 85%, are not among the
candidates at all, so no ranker on these candidates can exceed the candidate ceiling. The ranker
is trained only on customers who bought in the target week, which matches the evaluation but not a live
setting where you do not know who will buy. Hyperparameters were not tuned, so this is not the best this model could do.

## Experimentation A: a real A/B test (Day 6)

**Data and why.** [Cookie Cats](https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats)
is a public mobile-game A/B test with one row per user: the arm they were assigned to, whether they
came back 1 and 7 days after install, and how many rounds they played (in the first 14 days, according to the dataset description).
The change under test: the first forced waiting gate moved from level 30 (control, `gate_30`) to level 40
(`gate_40`). I picked it because it has user-level assignment and outcomes, real (not simulated) users,
two binary metrics and one heavy-tailed metric, which covers what the analysis steps need. H&M has no
experiment data. Script: `python scripts/download_ab_data.py` then `python scripts/ab_cookie_cats.py`;
numbers in `results/ab_cookie_cats.json`. Control is A, treatment is B, differences are B minus A.

**Sample.** 90,189 users: 44,700 in `gate_30`, 45,489 in `gate_40`. No missing values, no duplicate
user ids. 4.4% of users played zero rounds; they are kept (everyone assigned is analysed, whether or
not they engaged).

**Step 1: sample ratio mismatch (SRM).** *Why first:* if the arms do not have the size the design says,
assignment or logging is broken and every later number is suspect. A chi-square test of the arm sizes
against an even split (I assume 50/50; the dataset does not say) gives chi2 = 6.90, **p = 0.0086**. The
arms differ by 789 users (49.6% vs 50.4%). Under the common alarm threshold of p < 0.001 this passes, at
0.05 it fails, so it is a warning and not a verdict. I cannot investigate the cause with this data (no
timestamps or assignment log), so the later results carry this caveat: if users were dropped from one arm
for a reason related to the outcome, the comparison is biased.

**Step 2: retention, difference in proportions.** *Why this test:* both metrics are yes/no per user, and
the arms are independent, so the natural comparison is the difference in rates with a z-test. The
p-value uses the pooled variance (it describes the world where both arms share one rate, the null);
the confidence interval uses the unpooled variance (it describes the difference actually observed).
Both tests are two-sided. The bootstrap interval resamples users with replacement, which for a 0/1
outcome is a binomial draw per arm, and does not rely on the normal approximation.

| Metric | gate_30 | gate_40 | Difference (B - A) | 95% CI (analytic) | 95% CI (bootstrap) | p-value |
|---|---|---|---|---|---|---|
| 1-day retention | 44.82% | 44.23% | -0.59 pp (-1.3% relative) | [-1.24, +0.06] pp | [-1.23, +0.04] pp | 0.074 |
| 7-day retention | 19.02% | 18.20% | -0.82 pp (-4.3% relative) | [-1.33, -0.31] pp | [-1.32, -0.31] pp | 0.0016 |

7-day retention is lower with the later gate, and the result survives a correction for looking at two
retention metrics (Bonferroni, alpha 0.025: interval [-1.40, -0.24] pp still excludes zero). 1-day
retention is not significant. The analytic and bootstrap interval ends agree to within 0.02 pp, which is what
I expect at this sample size.

**Step 3: statistical vs. practical significance.** A p-value says whether an effect is distinguishable
from zero, not whether it matters. I assumed that a change of **0.5 percentage points** in retention is the
smallest worth acting on. This threshold is my assumption, not from the data, and should come from the
business (what a retained player is worth). Against it:
- **7-day retention:** statistically significant, and the interval [-1.33, -0.31] pp spans sizes from
  smaller than the threshold to more than twice it. So the direction is clear (the later gate hurts) but
  the size is not pinned down.
- **1-day retention:** not significant, and the interval [-1.24, +0.06] pp does not rule out a
  practically relevant drop. "Not significant" here means "not enough data", not "no effect".

**Step 4: game rounds (a heavy-tailed metric).** The mean is 52.46 rounds (`gate_30`) vs 51.30
(`gate_40`), the median 17 vs 16. A Welch t-test gives p = 0.38, a Mann-Whitney test p = 0.050, and the
bootstrap interval for the mean difference is [-4.05, +0.93]. The raw mean difference of -1.16 is almost
entirely one user: the largest value in `gate_30` is 49,854 rounds, while the next largest is 2,961. Without
that single user the mean difference is -0.04, and with all values capped at the 99.9th percentile
(1,074 rounds) it is -0.10 (interval [-1.35, +1.16]). Conclusion: no evidence that the gate position changes
how much people play, and means of this metric should not be trusted without trimming. I did not
check whether 49,854 is a bot or a logging error.

**Step 5: does the pipeline give honest p-values? (A/A check on the real users.)** I shuffled the arm
labels 20,000 times (so there is no real difference by construction) and counted how often the test
rejects at 0.05: 5.07% for both retention metrics. The test is calibrated on this data. This shows the
procedure is sound, not that this particular result is.

**Step 6: what could this experiment detect?** With about 44,700 users per arm, 80% power and alpha 0.05,
the smallest detectable change is **0.93 pp for 1-day retention and 0.74 pp for 7-day retention**. The
observed 1-day change (0.59 pp) is below what the test could reliably see, which explains the
non-significant result. To detect 0.5 pp in 7-day retention you would need about 97,700 users per arm
(more than twice this experiment); to detect 1.0 pp, about 24,700.

**What I would conclude.** Keep the gate at level 30: moving it to 40 lowered 7-day retention by
0.82 pp (95% CI 0.31 to 1.33 pp), did not raise engagement (rounds), and the 1-day result points the same
way without being significant. This is the retention side only; the dataset has no revenue or ad data,
and a later gate may have monetisation benefits that this test cannot show.

**Limitations.**
- Primary metric: I treated 7-day retention as the main metric, but I chose that after seeing both
  results; there was no pre-registration. With two retention metrics and one engagement metric, the
  Bonferroni check above is the minimum correction.
- The SRM p-value of 0.0086 is unexplained, and the 50/50 design is my assumption.
- No timestamps, so no check for novelty effects or time trends; no user covariates, so no CUPED (that
  comes in Day 7 on H&M data) and no segment analysis; rounds played is measured after assignment, so
  splitting users by it would bias the comparison.
- Retention is only measured at day 1 and day 7; longer-term effects are unknown. Users may share devices
  and are assumed independent.

## Experimentation B: designing the test for the recommender (Day 7)

H&M has no experiment data, so this part is a simulation calibrated on real H&M behaviour: the
customers and their purchases are real, and only the random assignment and the treatment effect are
simulated. Script: `python scripts/experiment_sim.py` (a few minutes; it scans the full transactions
file once to count eligible customers); numbers in `results/experiment_sim.json`.

**Setup.** *Population:* customers who bought something in the 56 days before 2020-08-12 (the covariate
window, 2020-06-17 to 2020-08-11). *Outcome:* what they do in the next 14 days (2020-08-12 to 2020-08-25).
Both windows are before the validation and test weeks. In the 10% customer sample that is **43,182
customers**; in the full data it is **428,895**. Two metrics: **purchase rate** (bought anything in 14 days) and
**revenue per user** (sum of prices; prices in this dataset are scaled, so these are price units, not
currency). The pre-period spend is the CUPED covariate and is only built from rows before the window starts
(a test checks that purchases in or after the window never change it).

**What the population looks like.**

| Quantity | Value |
|---|---|
| Purchase rate in 14 days | 20.2% |
| Revenue per user: mean / standard deviation | 0.0231 / 0.0784 (coefficient of variation 3.4) |
| Correlation of 14-day revenue with pre-period spend | 0.32 |
| Correlation of 14-day purchase with pre-period spend | 0.17 |

Revenue per user is very skewed: four in five customers spend nothing in the window and the rest vary a
lot. That is why revenue is a much noisier metric than purchase rate.

**Sample size and detectable effect.** Users per arm for 80% power, alpha 0.05, two-sided, by relative
uplift of the metric. The uplifts are assumptions; I do not know what a real recommender change would give.

| Relative uplift | Purchase rate | Revenue per user | Revenue per user, with CUPED |
|---|---|---|---|
| 1% | 622,248 | 1,807,119 | 1,627,690 |
| 2% | 156,135 | 451,780 | 406,923 |
| 5% | 25,253 | 72,285 | 65,108 |
| 10% | 6,424 | 18,072 | 16,277 |

Revenue needs about 2.9 times as many users as purchase rate for the same relative uplift. Whether a test
is feasible depends on the base. If the whole eligible base is split in two for one 14-day cohort:

| Population (users per arm) | Smallest detectable uplift, purchase rate | Smallest detectable uplift, revenue per user (CUPED) |
|---|---|---|
| This 10% sample (21,591) | 5.4% | 8.7% (9.1% without CUPED) |
| Full data (214,447) | 1.7% | 2.8% (2.9% without CUPED) |

Even with every eligible H&M customer in one test, a 2% revenue uplift is below what a 14-day test can
see (it needs 452,000 per arm, about 2.1 times the 214,447 available per arm). A 2% uplift in purchase rate
is detectable.

**Do the formulas hold? (simulation on the real customers.)** I drew customers with replacement from the
population (which stands in for a larger, similar population), split them into two arms, multiplied the
treated customers' revenue by 1 + uplift, and counted rejections.

| Setting | Naive | CUPED |
|---|---|---|
| No effect (A/A), 18,072 per arm, 1,000 simulations: false positives | 4.8% | 4.9% |
| 10% uplift at 18,072 per arm, 1,000 simulations: power | 76.5% | 79.7% |
| 5% uplift at 72,285 per arm, 500 simulations: power | 75.6% | 80.8% |

The false-positive rate is on target. Power for the naive test is a few points under the planned 80%: the
formula assumes both arms have the same variance, but multiplying treated revenue by 1.1 also scales their
standard deviation by 1.1, which by my arithmetic lowers the expected power to about 76%, in line with the
simulation. This is a limit of the textbook formula, not an error in the simulation.

**CUPED: what it actually gives here.** CUPED subtracts the part of the outcome predicted by pre-period
spend. Measured variance reduction: **9.9% for revenue** (the squared correlation, 0.32^2) and 2.8% for
purchase rate. Adding the pre-period order count as a second covariate did not reduce it further (9.9%).
In plain terms: about 10% fewer users for the same power, 65,108 instead of 72,285 per arm for a 5% revenue
uplift. The simulation agrees: the estimate stays unbiased (mean estimated uplift 0.1008 with CUPED vs 0.1007
without, for a true 0.10; 0.0501 vs 0.0499 for a true 0.05), its standard deviation falls (0.0387 to 0.0370 and
0.0191 to 0.0175), and power rises by 3 to 5 points. This is a modest gain, much smaller than the
variance reductions you may see quoted for metrics with strong week-to-week persistence. Spend in a short
window is mostly a question of whether a customer buys at all, and past spend says little about that.

**Peeking.** A/A tests (no true effect) on the real revenue data with 18,072 users per arm planned, where
the test is declared a win at the first look with p < 0.05. 3,000 simulations per row (about +-0.7 points):

| Looks (evenly spaced) | 1 (fixed horizon) | 2 | 5 | 10 | 14 (daily for 14 days) | 20 |
|---|---|---|---|---|---|---|
| False-positive rate | 4.9% | 8.4% | 13.1% | 19.0% | **21.5%** | 23.5% |

Looking daily and stopping at the first significant result turns a 5% error rate into about 21%. Fixing it with
14 looks, on the same 3,000 simulated experiments with a 10% true uplift for the power column:

| Rule | False positives (A/A) | Power at 10% uplift | Average share of planned sample used (with the uplift) |
|---|---|---|---|
| Fixed horizon, look once at the end | 4.3% | 76.0% | 100% |
| Peek daily at p < 0.05 | 21.5% | 85.1% (not a real gain; includes false wins) | 49.8% |
| Bonferroni (0.05 / 14 per look) | 1.9% | 50.1% | 79.3% |
| Pocock (constant boundary, z = 2.62) | 4.6% | 62.2% | 71.1% |
| O'Brien-Fleming style (z = 2.10 / sqrt(share of sample)) | 4.9% | 74.5% | 78.0% |

Bonferroni is too conservative for correlated looks, and Pocock costs 14 points of power. The O'Brien-Fleming
style rule keeps the 5% error rate, loses about 1.5 points of power compared with a fixed horizon (each
figure has simulation noise of about 0.8 points, so this is a small loss at most) and stops early on average, using 78% of the planned sample when the
effect is real. Boundaries were calibrated by simulating the null (Pocock and the O'Brien-Fleming constant),
and the Pocock constant for 5 looks matches the textbook value of 2.413 in a test. Limits of the simulation: users arrive evenly and
each one's full outcome is known at their look; in a real test, a customer enrolled on day 10 has not yet had
14 days to buy.

### How I would A/B test this recommender

1. **Unit and eligibility.** Randomise by customer id (a hash of the id and an experiment name, 50/50),
   so a customer always sees the same arm. Analyse everyone assigned, whether or not they looked at the
   recommendations (intent to treat). Eligible: customers with a purchase in the last 8 weeks, which is the
   population simulated above.
2. **Primary metric: 14-day purchase rate.** It is what a better ranking is meant to move, and it needs
   about 2.9 times fewer users than revenue per user (table above). With the full eligible base it can detect
   a 1.7% relative uplift; revenue per user can only detect 2.8%. This is a trade-off: purchase rate can rise
   through cheaper items, so revenue is a guardrail, not ignored.
3. **Guardrails** (each with a non-inferiority margin set before the test): revenue per user, average price of
   items bought and discount share (a margin check), returns rate (not in this dataset), the share of requests
   served by the popularity fallback, `/recommend` latency (p95) and error rate, and catalogue coverage
   (a ranker that recommends only the same bestsellers is a risk). Check the sample ratio on every run and
   alarm at p < 0.001. Report new vs. returning customers separately: the offline results show the ranker is
   weakest for customers with no history.
4. **Variance reduction.** CUPED on pre-period spend, expecting about 10% fewer users. Not more, based on
   the measurement above.
5. **Duration and sample size.** Fix both in advance. At least 14 days, so the outcome window covers two
   full weekly cycles, and never over a sale: the weekly counts in the EDA show a spike in June. Compute the
   sample from the smallest uplift worth shipping, not from the offline gain. If that uplift is 2% or less, plan on
   a longer test with more distinct customers, because 14 days of every eligible customer is not enough for
   revenue.
6. **Offline gain is not an online forecast.** The ranker beat the baseline by about 20% on MAP@12 offline. That does
   not say how much purchase rate or revenue will move, which is why the test is needed. I would not
   assume an uplift of any size.
7. **Peeking.** Either look once at the end, or pre-register a group-sequential plan with O'Brien-Fleming style
   boundaries (daily looks, about 1.5 points of power lost in the simulation, early stop when the effect is large).
   No ad-hoc daily checks of p-values.
8. **Rollout.** First an A/A run to check the pipeline and the variance; then a small ramp (1%, then 10%) watching only
   the guardrails, since those ramps are too small to say anything about the primary metric; then the 50/50 test.
   If it passes the pre-set rule (lower end of the primary CI above zero, guardrails within margin), roll out
   gradually and keep a 5% holdout for a few weeks to look for novelty effects and long-term drift. The popularity
   fallback and a kill switch stay in place.

**Limits of this analysis.** Effect sizes are assumed, and the multiplicative effect on revenue is one possible
model (a recommender may change which customers buy, not scale everyone's spend). The simulation draws with
replacement from 43,182 customers, which treats them as a sample from a larger population. Customers who did not buy in the 8 weeks
before the window are not in it, and non-buying visitors are not in this data at all. The
100% coverage of outcomes at every look is optimistic, as noted above.

## Layout

```
src/hm_recsys/metrics.py   MAP@k, recall@k, NDCG@k, per-segment evaluation
src/hm_recsys/data.py      DuckDB loading, subsampling, time split, leakage helper
src/hm_recsys/baselines.py popularity, age-group popularity, buy again, fallback filling
src/hm_recsys/candidates.py co-purchase, ALS, popularity and buy-again candidates, union, recall
src/hm_recsys/features.py  candidate rows and the 33 ranking features, labels, training weeks
src/hm_recsys/ranker.py    LightGBM LambdaRank training, top-k prediction, importance
src/hm_recsys/experiments.py SRM, proportion tests, bootstrap, sample size and power
src/hm_recsys/simulation.py experiment table, CUPED, power and peeking simulations
scripts/                   download, prepare, EDA, baselines
tests/                     unit tests on synthetic fixtures
```
