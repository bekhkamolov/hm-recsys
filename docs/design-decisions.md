# Design decisions

Part of [hm-recsys](../README.md). Every number here comes from the JSON files in `results/`.

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
| Sequential testing | 14 daily looks; Pocock and O'Brien-Fleming-style constants calibrated by simulating the null | Simulation-calibrated boundaries need no lookup tables for other numbers of looks; the Pocock constant for 5 looks matches the textbook value in a test. |
| Serving split | Candidates and features precomputed in batch, ranker scored per request | Candidate generation is slow and needs full history; scoring about 290 rows is cheap. Cost: recommendations are as fresh as the last snapshot. |
| Artifact format | LightGBM text model, parquet of candidate rows, JSON for popularity and metadata | Portable and inspectable; the service loads it with LightGBM, pandas and numpy only. |
| Fallback | Global popularity of the snapshot's last 7 days | The best non-personalised list from Day 2; used for any customer not in the table. |
| Snapshot date | 2020-09-09 (validation-week start) | Kept the test week untouched until the final evaluation. |
| Packaging | Base install = serving runtime; training, serving, tracking and dev in extras | A slim image and a faster build; a test checks the API imports no training code. |
| Tracking backend | MLflow with SQLite (`mlflow-skinny` + SQLAlchemy + Alembic) | MLflow 3.x refuses its file store by default; skinny is enough to log and read runs. |
| Deployed model | Synthetic demo model baked into the image | The H&M data licence forbids redistribution, and per-customer rows derived from it should not be in a public repo or image. Same code path as the H&M model. |
| Drift statistic | PSI, 10 reference-quantile bins, missing as its own bin, exact-value bins for few-valued features | Standard in credit and ranking monitoring; one number per feature that is easy to alert on. |
| Drift thresholds | 0.1 and 0.25 | Convention. The noise floor (mean 0.0017, max 0.0096) shows they are far above sampling noise here, but they are not tuned to performance. |
| Noise floor | PSI between two halves of one week split by customer | A row-level split puts the same customer in both halves and hides the noise (a test checks this). |
| Retraining evidence | Models trained 0 to 3 weeks earlier, 3 seeds each, scored on the validation week | Measures what staleness costs instead of inferring it from PSI. One week and one sample, so differences under 0.0006 are noise. |
| Live monitoring | In-process `/stats` (fallback share, scoring time), no customer ids | Cheap and privacy-safe; counters reset on restart and are per process. |
| Reference baseline | Buy again + global popularity | The age-group variant did not beat it, so the simpler one is kept (see [baselines](data-and-baselines.md)). |
