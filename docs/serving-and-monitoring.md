# Serving, deployment and monitoring

Part of [hm-recsys](../README.md). Every number here comes from the JSON files in `results/`.

## Serving, tracking and CI (Day 8)

```
 batch, offline (scripts/build_serving.py)                online (Docker, hm_recsys.api)
 ┌───────────────────────────────────────────┐          ┌─────────────────────────────────────┐
 │ transactions ─> candidates (4 sources)    │          │ GET /recommend?customer_id=...&k=12 │
 │              ─> 33 features per pair      │  files   │   known customer: LightGBM scores   │
 │ train LightGBM ranker (4 weeks)           ├─────────>│   its ~290 precomputed candidates   │
 │ write: model.txt, candidates.parquet,     │ /app/    │   unknown customer: popularity      │
 │        popularity.json, meta.json         │ artifacts│ GET /health: ready, snapshot date   │
 └───────────────────────────────────────────┘          └─────────────────────────────────────┘
```

**Why split it this way.** Generating candidates (ALS, co-purchase) and computing features needs the
full purchase history and is slow, so it is done in batch. Scoring about 290 candidates with a
LightGBM model is cheap (a whole request takes about 1.5 ms, see below), so the ranker runs per request. The service therefore needs
only the model and one table of precomputed rows, no purchase history, DuckDB or `implicit`. The cost
is freshness: recommendations reflect the snapshot, not what the customer did an hour ago.

**The endpoint.** `GET /recommend?customer_id=<id>&k=12` (k from 1 to 50) returns
`{"customer_id", "source", "article_ids", "scores"}`. `source` is `"ranker"` for a customer in the
precomputed table and `"popularity_fallback"` otherwise (unknown customer, or one without a table row),
with the global most popular articles of the snapshot's last 7 days and `scores: null`. If a known customer
has fewer than k candidates, the remaining slots are filled from popularity and have a `null` score. A missing or
empty `customer_id` or a k outside 1-50 returns 422. `GET /health` returns `ready`, the snapshot date, the
number of customers and features. Without artifacts the service starts, reports `ready: false` and answers
503 on `/recommend`, so a missing model is visible and does not crash the container.

**Run it.**
```bash
# with the H&M sample prepared (a few minutes; writes ./artifacts, about 16 MB for 2,000 customers)
python scripts/build_serving.py --max-customers 2000
ARTIFACT_DIR=artifacts uvicorn hm_recsys.api:create_app --factory
curl "localhost:8000/recommend?customer_id=<an id from artifacts/meta.json>"

# no data needed: synthetic demo artifacts
python scripts/make_demo_artifacts.py --out artifacts-demo
docker build -t hm-recsys .
docker run -p 8000:8000 -v "$PWD/artifacts-demo:/app/artifacts:ro" hm-recsys
```

**The snapshot is as of 2020-09-09** (the start of the validation week): the model is trained on the 4
weeks before it, and the candidate rows use purchases before it. The test week is not used for this snapshot
(it was evaluated separately, once; see the [final evaluation](final-evaluation.md)). A production system would rebuild this weekly. The demo table holds 2,000 of
the 59,640 sampled customers (a reproducible sample of customers with a purchase in the 8 weeks before the
snapshot, chosen without looking at the following week); everyone else gets the popularity fallback.

**Checked, on the real artifacts** (local, one process, a Python HTTP client on the same machine, so
this is not a load test): the model in the artifact scores MAP@12 0.0279, recall@12 0.0556, NDCG@12 0.0407 on
the full validation week, identical to seed 0 of the table in the [ranker document](ranker.md), so the serving pipeline reproduces
the research result. A ranked request takes a median 1.5 ms (p95 2.0 ms) with about 290 candidates per
customer; the fallback about 0.9 ms. The feature matrix is 579,098 rows x 33 float32 columns, about 76 MB in memory.

**Tests** (all on synthetic data, no network):
- The served top 12 equals the offline model's top 12 for every served customer (parity).
- Appending purchases from the prediction week does not change the precomputed candidates or the
  popularity list (no leakage through the artifact path).
- Known customer, unknown customer (fallback), response shape, k limits, 422s and 503 without artifacts.
- The API module imports none of the training code.

**Docker and CI.** The image (`python:3.11-slim`, non-root user, health check, listens on `$PORT` as
Render sets it) is built with the `serve` extra only: no DuckDB, `implicit`, scikit-learn or MLflow (SciPy is still
installed, because LightGBM depends on it); the model artifacts are mounted, not baked in. The GitHub Actions workflow has two jobs. `test`: ruff and pytest.
`docker`: builds demo artifacts from synthetic data, builds the image, starts it, and checks `/health`, that
it runs as non-root, that a known customer is ranked by the model, that an unknown customer gets the fallback,
that a bad request gives 422, that a container without artifacts is up but not ready (503), and that it listens
on `$PORT`. I could not run Docker on my machine, so what I verified locally is everything except the image
itself: a clean environment with only the `serve` extra installed, the same `uvicorn` command, and the
checks above. The image build is covered by the CI run.

**MLflow tracking.** Every experiment script logs its run when it finishes, and `scripts/track_results.py`
backfills existing `results/*.json`. A run has the sample description as parameters, every number as a metric
(for example `full_model_seeds.0.map_12`) and the JSON as an artifact; `build_serving.py` also logs the
model file and its validation metrics. Runs go to a local SQLite file (`mlflow.db`) because MLflow 3.x
has put its file store into maintenance mode and refuses it by default; both `mlflow.db` and `mlruns/` are
ignored by git. The project depends on the lighter `mlflow-skinny`, which is enough to log and to read runs back
(the tests do). To browse them in the web UI (`mlflow ui --backend-store-uri sqlite:///mlflow.db`) you
need the full `mlflow` package; I have not installed or tried it here. Without MLflow installed
the scripts still run and print a note.

**Limits.** The precomputed table covers a sample of customers, not everyone. The snapshot is one fixed
date, with no scheduled rebuild yet. Features are not recomputed per request, so a purchase made after the
snapshot does not change a customer's list. The latency figure is local and single-process. There is no
authentication or rate limiting on the endpoint. Monitoring and deployment are Day 9.

## Deployment and monitoring (Day 9)

### Deployment (Render)

`render.yaml` is a Render Blueprint for a free web service in Frankfurt that builds the same
`Dockerfile`, health-checks `/health`, and redeploys only after CI passes (`autoDeployTrigger: checksPass`),
the same setup as the support agent. To deploy: in the Render dashboard choose New, then Blueprint, select this
repository and apply; no secrets are needed.

**What is deployed is a synthetic demo model, not the H&M model.** The image contains a small model trained on
synthetic data (`deploy/demo-artifacts`, made by `scripts/make_demo_artifacts.py`, 289 fake customers `c0001`
onwards) and `ARTIFACT_DIR` points at it. I did this on purpose: the H&M data is licensed for the competition and
must not be redistributed, and per-customer feature rows derived from it do not belong in a public repository
or image. The code path, the API and the CI checks are identical to the H&M version, which you can run
locally with `scripts/build_serving.py` (see above). A test fails if the committed demo artifacts stop matching
the current feature set. Check a running service with
`python scripts/smoke_test.py https://<service>.onrender.com`. Render's free tier stops idle services,
so the first request can be slow; the script waits for the service to wake up. The deployed service at
https://hm-recsys.onrender.com passes all four smoke-test checks, and returns the same scores for `c0001` as the
same demo model run locally. A known-customer request took 125 ms from my machine, which is mostly network time
(the model itself takes about 1 ms, see `/stats`).

### Monitoring

**Operational numbers.** `GET /stats` returns request counts by source (ranker vs popularity fallback), the fallback
share, and the mean and 95th percentile scoring time over the last 1,000 requests. It keeps no customer ids and counts
since the process started. A rising fallback share means customers are missing from the precomputed table (the snapshot
is getting old or the customer base moved). `GET /health` shows which snapshot is loaded.

**Data drift: population stability index (PSI).** PSI compares one variable's distribution in a reference sample
(here the data the model was trained on) with a current sample (here the candidates for the week being served):
ten bins from the reference quantiles, missing values as their own bin, and the sum of (current share - reference
share) x ln(current share / reference share). Variables with 10 or fewer distinct values are binned by value. The usual
reading is below 0.1 stable, 0.1 to 0.25 moderate, above 0.25 major. Those cut-offs are convention, not validated
on this data. Code: `src/hm_recsys/monitoring.py`; experiment: `python scripts/drift_check.py` (roughly 10 to 15 minutes);
numbers in `results/drift_val.json`. The reference is the 4 training weeks of the headline model (7.34M candidate
rows); the current sample is the validation-week snapshot (1.81M rows, 2020-09-09). The test week is not used.

| Quantity | Value |
|---|---|
| Mean PSI over the 33 features, training data vs. validation week | 0.089 |
| Features at 0.1 or more / at 0.25 or more | 5 / 2 |
| PSI from sampling noise alone (one week split by customer): mean / largest | 0.0017 / 0.0096 |
| PSI of the model score (last training week vs. validation week) | 0.0043 (stable) |

Real shifts are 10 to 100 times the noise level, so the 0.1 cut-off is far above noise at this sample size.
Mean PSI by feature group: article 0.153, customer 0.111, age-group popularity 0.018, customer x article history
0.005, ALS / co-purchase scores 0.003, source ranks 0.0005. The features the ranker relies on most (history,
scores, ranks, see the ablation) are the stable ones. The two major shifts are
`a_days_since_first` (1.13) and `c_days_since_first` (0.78), followed by product type (0.19), department (0.14) and
7-day article popularity (0.12). **The two largest are drift by construction, not behaviour:** they measure
days since first purchase inside a fixed data window that starts on 2020-06-03, so they grow every week for everyone.
A production alert should ignore such window-relative features or recompute them over a fixed horizon. The model
score barely moves (mean -1.73 to -1.80, standard deviation 1.62 to 1.64) even though several features do.

**How fast does it drift?** PSI of each earlier week against the validation week, mean over features (features
at 0.1 or more in brackets): 1 week earlier 0.167 (4), 2 weeks 0.207 (6), 3 weeks 0.281 (8), 4 weeks 0.389 (11),
5 weeks 0.383 (12), 6 weeks 0.419 (14), 7 weeks 0.487 (18). It rises with age, with one small dip at 5 weeks.

**What does staleness actually cost?** PSI says the data moved, not that predictions got worse, so I measured it.
I trained the headline model on the 4 weeks ending `s` weeks before the validation week (3 seeds each, same pipeline)
and scored the validation week:

| Model trained | Training weeks start | MAP@12 (3 seeds, mean) | recall@12 | NDCG@12 | Mean PSI vs. validation | Features >= 0.1 |
|---|---|---|---|---|---|---|
| Right before validation | 2020-08-12 | 0.0273 (0.0274 / 0.0267 / 0.0276) | 0.0536 | 0.0395 | 0.089 | 5 |
| 1 week earlier | 2020-08-05 | 0.0276 (0.0273 / 0.0281 / 0.0273) | 0.0539 | 0.0399 | 0.144 | 7 |
| 2 weeks earlier | 2020-07-29 | 0.0274 (0.0276 / 0.0271 / 0.0276) | 0.0558 | 0.0402 | 0.185 | 9 |
| 3 weeks earlier | 2020-07-22 | 0.0261 (0.0265 / 0.0262 / 0.0257) | 0.0511 | 0.0377 | 0.354 | 11 |

A model that is 1 or 2 weeks old is no worse than a fresh one (differences of 0.0001 to 0.0003 are within noise). At 3 weeks
the model is about 4% worse in relative terms (MAP@12 0.0261 vs. 0.0273; recall@12 and NDCG@12 about 5% lower), and
all three seeds are below every seed of the newer models. Read this with care:
- **Noise.** The same fresh model trained with a different order of customers in the training file scored 0.0273 to
  0.0276 (3-seed means), and the headline run, which used a different but equivalent data path, scored 0.0279. I could not
  trace the 0.0003 to 0.0006 gap to a single cause, so treat differences below about 0.0006 as noise; this is larger
  than the seed-to-seed spread quoted in the [ranker document](ranker.md).
- **Confounded.** Older models were trained on earlier weeks, which also have less purchase history behind their
  features, so staleness and shorter history cannot be separated here.
- **One validation week**, one 10% customer sample, three seeds.
- **PSI is a weak predictor in this data.** Mean PSI crossed 0.1 at 1 week and performance did not drop until 3 weeks,
  when mean PSI was 0.35 and 6 features were major.

**When to retrain.** What the measurements support: weekly retraining is more than enough, since 1 to 2 weeks of age
cost nothing measurable and 3 weeks cost about 4%. It is also cheap at this scale: building a week of candidates takes
about 15 seconds and training the model about 20 seconds on the 10% sample (not measured on the full data, which is
about 10 times larger). What I would add, as a proposal that these results neither prove nor disprove:
1. Retrain on a schedule (weekly), regardless of drift.
2. Retrain early if the model score PSI is above 0.1, or PSI is major for features in the history, score or rank
   groups, since those are what the model uses and they were stable here.
3. Watch `/stats`: investigate a fallback share well above its normal level.
4. Once a week's purchases are known, compute that week's MAP@12 for the served model and compare it with the
   previous weeks; a relative drop of more than about 5% (the size of the 3-week effect) triggers a retrain and a
   look at the data. Labels arrive a week late, so this is the slowest check and the only one that measures quality.
Alert thresholds for the first three are conventions and would need calibrating on a live system. Performance
monitoring over time was not run here: it needs several weeks after the snapshot, and only one (the test week) is
left, which was reserved for the final evaluation.
