# hm-recsys

Product recommendations for a fashion retailer on the public
[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
dataset (real transactions), plus a module on how to measure a recommender change online.

**What is here.** A recommender (four candidate sources, then a LightGBM LambdaRank ranker with 33 features) and the
simple baselines it has to beat; evaluation with strict time-based splits and leakage tests; a real A/B-test
analysis; simulations on H&M purchase behaviour for sample size, CUPED and peeking; a FastAPI service in Docker with a
popularity fallback, MLflow tracking, drift monitoring and CI; and a deployed demo. Only numbers that were actually
measured are reported, including the bad ones.

**Headline result.** On the held-out test week (2020-09-16 to 2020-09-22), evaluated once, for the 6,972 customers in a
10% customer sample who bought that week, predicting 12 articles each:

| Model | MAP@12 | recall@12 | NDCG@12 |
|---|---|---|---|
| Popularity (last 7 days) | 0.0112 | 0.0298 | 0.0191 |
| Buy again + popularity | 0.0267 | 0.0561 | 0.0387 |
| **LightGBM ranker** (mean of 3 seeds) | **0.0306** | **0.0635** | **0.0446** |

The ranker is 14.9% better than "buy again" on MAP@12 (paired bootstrap, seed 0: +0.0038, 95% interval +0.0019 to
+0.0057), less than its 19.6% gain on the validation week. It is clearly better for customers with purchase history and
**worse than plain popularity for customers with none**. Absolute values are low and not comparable to the Kaggle
leaderboard: it is a 10% sample, and the candidates reach only 17% of the purchased articles, which caps any ranker
built on them. Details: [final evaluation](docs/final-evaluation.md).

**Live demo: https://hm-recsys.onrender.com** (a free instance that sleeps when idle, so the first request can be
slow; it serves a small synthetic demo model, not the H&M model, because of the data licence).
Try [`/recommend?customer_id=c0001`](https://hm-recsys.onrender.com/recommend?customer_id=c0001), an unknown id for the
popularity fallback, [`/health`](https://hm-recsys.onrender.com/health),
[`/stats`](https://hm-recsys.onrender.com/stats) and the interactive [`/docs`](https://hm-recsys.onrender.com/docs).

## Documentation

| Document | What is in it |
|---|---|
| [Data, baselines and candidates](docs/data-and-baselines.md) | The dataset, evaluation protocol, exploratory analysis, popularity and "buy again" baselines, candidate sources and their recall |
| [Ranker](docs/ranker.md) | LightGBM LambdaRank, the 33 features, ablation by feature group, importance, how much training data helps |
| [Experimentation](docs/experimentation.md) | A real A/B test (Cookie Cats), H&M-calibrated simulations (sample size, CUPED, peeking) and a plan for A/B testing this recommender |
| [Serving and monitoring](docs/serving-and-monitoring.md) | The API, Docker image, CI, MLflow, Render deployment, drift checks and the retraining analysis |
| [Final test-week evaluation](docs/final-evaluation.md) | The one test-week run: tables by model, segment and age, validation vs. test |
| [Design decisions](docs/design-decisions.md) | Every choice with the reason for it |

## Results at a glance

| Question | Answer |
|---|---|
| Does a ranker beat simple baselines? | Yes. Test: MAP@12 0.0306 vs 0.0267 (buy again) vs 0.0112 (popularity). Validation: 0.0279 vs 0.0233 vs 0.0053. |
| Does it help every customer? | No. With no purchase history in the window it is worse than popularity on the test week (0.0068 vs 0.0113). |
| How much of the purchases can candidates reach? | 14.6% (validation) and 16.9% (test), which caps any ranker. |
| What in the ranker matters? | Customer x article history most (removing it costs 0.0021 MAP@12). No single feature group beats the baseline alone. |
| Does more training data help? | Up to a point: MAP@12 0.0246 (1 week), 0.0257 (2), 0.0279 (4), 0.0281 (6). |
| Real A/B test (Cookie Cats) | The later gate lowered 7-day retention by 0.82 pp (95% CI 0.31 to 1.33 pp, p = 0.0016); 1-day not significant. An unexplained sample-ratio warning (p = 0.0086). |
| What can an H&M test detect? | With every eligible customer split 50/50 for 14 days: a 1.7% relative uplift in purchase rate, 2.8% in revenue per user. CUPED saves about 10% of users. |
| What does peeking cost? | 14 daily looks at p < 0.05 turn a 5% false-positive rate into 21.5%; an O'Brien-Fleming-style boundary holds it at 4.9%. |
| How fast is serving? | Median 1.5 ms per request locally (p95 2.0 ms), scoring about 290 candidates per customer. |
| When to retrain? | Weekly is more than enough: a model 1 to 2 weeks old is no worse, 3 weeks old about 4% worse (one validation week). |

## Architecture

```
 H&M CSV (31M rows) --DuckDB--> 16 weeks, 10% of customers --> parquet     (raw data never committed)
                                          |
        for each prediction week T, using only purchases before T (a test checks this)
                                          v
   candidates, up to 100 each:  buy again | co-purchase | ALS | popularity   (about 290 per customer)
                                          v
   33 features per (customer, article): customer, article, customer x article history,
                                        age-group popularity, ALS / co-purchase scores, source ranks
                                          v
   LightGBM LambdaRank, trained on the 4 weeks before T  -->  top 12 per customer
        |                                                        |
        | batch artifacts (model + candidate rows)               +--> offline metrics: MAP@12, recall, NDCG
        v                                                             validation week, test week (once)
   FastAPI /recommend, /health, /stats  (Docker, Render, CI smoke test)
     known customer -> LightGBM scores precomputed candidates; unknown -> popularity fallback
        |
        +--> MLflow runs, PSI drift and staleness checks, /stats (fallback share, scoring time)

 Experimentation (separate from the model): Cookie Cats A/B analysis; simulations on H&M behaviour
 for power, CUPED and peeking; a plan for A/B testing this recommender.
```

## Trade-offs: accuracy, complexity and latency

| Approach | Test MAP@12 | What it needs | Cost to run |
|---|---|---|---|
| Popularity | 0.0112 | One list per week | Almost nothing (the fallback path takes about 0.9 ms); no personalisation |
| Buy again + popularity | 0.0267 | A customer's own purchase history | Almost nothing |
| LightGBM ranker | 0.0306 (+14.9% over buy again) | Four candidate sources, 33 features, 7.4M training rows, a weekly batch rebuild, an artifact table, drift monitoring | About 1.5 ms per request locally plus a batch pipeline |

- **Accuracy vs complexity.** The ranker adds about 0.004 MAP@12 over "buy again" on the test week and brings a whole
  pipeline with it. Whether that is worth it depends on what an extra relevant item is worth, which only an online
  test can say. Parts of the pipeline buy little: removing ALS / co-purchase scores or age-group popularity costs
  about 0.0005 MAP@12, source ranks nothing measurable, and co-purchase is the weakest candidate source. These are
  candidates for simplification; I did not test removing them together.
- **Accuracy vs latency.** Latency is not the constraint (a whole request takes about 1.5 ms locally). Candidate
  generation and features are too slow and need full history, so they run in batch, and recommendations are only as
  fresh as the last snapshot.
- **Cost at scale.** On the 10% sample, a week of candidates takes about 15 s and training about 20 s. The full
  31M-row data, about ten times larger, was not measured.

## Limitations

- **Sample and scope.** Last 16 weeks, a 10% customer sample, and only customers who bought in the target week are
  evaluated. Compare rows, not levels; levels are not comparable with the Kaggle leaderboard.
- **Two weeks, large week-to-week variation.** Popularity alone scores 0.0053 on validation and 0.0112 on test. The
  ranker's gain was 19.6% on validation and 14.9% on test; the test figure is the honest one, since the number of
  training weeks was chosen on the validation week. Differences below about 0.0006 MAP@12 are within noise.
- **Reach.** 83% of purchased articles are not in the candidate set on the test week.
- **Offline only.** No online evidence; the uplifts in the experiment-design section are assumptions.
- **No history.** The ranker is worse than popularity for customers with no history (a post-hoc observation from the
  test week, so nothing was changed because of it).
- **Not used or not done.** No images, product text, sessions, discounts or stock; no hyperparameter search.
- **Deployment.** The public demo serves a synthetic model; the H&M artifact table covers 2,000 customers and is a
  fixed snapshot; the API has no authentication or rate limiting.
- **Experiments.** Cookie Cats has no timestamps or covariates and an unexplained sample-ratio warning; the H&M
  simulations assume every outcome is known at every look and a multiplicative treatment effect.

## Quick start

```bash
pip install -e ".[dev]"
pytest -q                     # runs on synthetic fixtures, no network or data download
ruff check . && ruff format --check .

# try the API locally, no data needed
python scripts/make_demo_artifacts.py --out artifacts-demo
ARTIFACT_DIR=artifacts-demo uvicorn hm_recsys.api:create_app --factory

# or in Docker: the image has the demo model baked in
docker build -t hm-recsys .
docker run -p 8000:8000 -e ARTIFACT_DIR=/app/demo-artifacts hm-recsys
```

**Reproduce the research** (needs the H&M data, a Kaggle token and a few GB of free memory; the Data section of the
[data document](docs/data-and-baselines.md) has the download steps). Each script writes a JSON file to `results/`,
which is what every table in these documents is taken from:

| Step | Command | Output |
|---|---|---|
| Subsample | `python scripts/prepare_data.py --weeks 16 --customer-fraction 0.1` | `data/processed/` |
| Baselines, candidates | `python scripts/baselines.py`, `python scripts/candidates.py` | `results/baselines_val.json`, `candidates_val.json` |
| Ranker | `python scripts/ranker.py --train-weeks 4` (about 15 min) | `results/ranker_val.json` |
| A/B test analysis | `python scripts/download_ab_data.py`, `python scripts/ab_cookie_cats.py` | `results/ab_cookie_cats.json` |
| Experiment simulations | `python scripts/experiment_sim.py` | `results/experiment_sim.json` |
| Drift and staleness | `python scripts/drift_check.py` (10 to 15 min) | `results/drift_val.json` |
| Serving artifacts | `python scripts/build_serving.py` | `artifacts/` (not committed) |
| Final evaluation | `python scripts/final_eval.py --target val` (dry run), then `--target test` (once) | `results/final_eval_*.json` |

Dependencies are split into extras: the base install is what the serving image needs; `.[train]`, `.[serve]` and
`.[tracking]` add the training, API and MLflow pieces; `.[dev]` installs all of them plus pytest and ruff.

## Key design decisions

The full table, with the reason for every choice, is in [design decisions](docs/design-decisions.md).

| Decision | Choice | Why |
|---|---|---|
| Split | Last 7 days = test, 7 days before = validation, the rest = train | Predict the next week from the past; random splits would leak. |
| Leakage guard | Features only through `history_before(df, cutoff)`; tests append future purchases and check nothing changes | Features for a week may only use earlier data. |
| Reference baseline | Buy again + popularity | The age-group variant did not beat it, so the simpler one stays. |
| Ranker training | Candidates and features rebuilt as of each of the 4 weeks before the target week, labelled with that week | The same procedure at train and predict time. |
| Model settings | Fixed in advance, untuned, no early stopping | The validation week is not used to fit the model. |
| Serving split | Candidates and features in batch, ranker per request | Generation is slow and needs full history; scoring is cheap. |
| Deployed model | Synthetic demo model baked into the image | The H&M licence forbids redistributing the data or per-customer rows derived from it. |
| Drift statistic | PSI with a noise floor from a by-customer split | One alertable number per feature, with a reference for what noise looks like. |

## Layout

```
src/hm_recsys/   metrics, data, baselines, candidates, features, ranker (research code)
                 experiments, simulation (A/B statistics and simulations)
                 artifacts, api, monitoring, tracking, demo (serving, drift, MLflow)
scripts/         one script per stage; each writes its JSON to results/
docs/            the per-stage write-ups
results/         the JSON behind every table
deploy/          the synthetic demo model baked into the image
tests/           unit tests on synthetic fixtures (no network, no data download)
Dockerfile, render.yaml, .github/workflows/ci.yml
```
