# Final test-week evaluation

Part of [hm-recsys](../README.md). Every number here comes from the JSON files in `results/`.

## Final test-week evaluation (Day 10)

The held-out test week, 2020-09-16 to 2020-09-22, was evaluated once, with `python scripts/final_eval.py --target test`
(results in `results/final_eval_test.json`). What was fixed before looking at it: the three models (popularity,
buy again + popularity, the LightGBM ranker); the ranker's settings (the headline settings from the [ranker document](ranker.md), no
tuning, no early stopping); that it is retrained on the 4 weeks that end right before the target week (here 2020-08-19
to 2020-09-15, so the validation week is a training week and the test week never is); and that all three seeds are
reported. Before that, the same script ran on the validation week as a dry run and reproduced every validation number
in the [ranker document](ranker.md) exactly, so the code path is the one already documented (`results/final_eval_val.json`).
One thing I did look at before this: the exploratory statistics (weekly purchase counts, repeat rate, article
concentration) were computed over all 16 weeks, including the test week. They did not feed into any model or choice.

**Sample:** the 10% customer sample, 6,972 customers who bought in the test week (21,715 purchased articles). Training
rows for the ranker: 7,421,690 candidate rows, 14,085 purchased.

| Model | MAP@12 | recall@12 | NDCG@12 |
|---|---|---|---|
| Popularity (last 7 days) | 0.0112 | 0.0298 | 0.0191 |
| Buy again + popularity | 0.0267 | 0.0561 | 0.0387 |
| LightGBM ranker, seed 0 | 0.0305 | 0.0628 | 0.0446 |
| LightGBM ranker, seed 1 | 0.0312 | 0.0641 | 0.0450 |
| LightGBM ranker, seed 2 | 0.0303 | 0.0637 | 0.0444 |
| **Ranker, mean of 3 seeds** | **0.0306** | **0.0635** | **0.0446** |

Paired bootstrap over customers (2,000 resamples) of the MAP@12 difference, ranker minus "buy again": seed 0 +0.0038
(95% interval 0.0019 to 0.0057), seed 1 +0.0045 (0.0026 to 0.0064), seed 2 +0.0036 (0.0017 to 0.0054); the difference
was above zero in every resample. As before, that interval only covers which customers bought that week.

**Validation vs. test.**

| Model | MAP@12 validation | MAP@12 test | recall@12 validation | recall@12 test |
|---|---|---|---|---|
| Popularity | 0.0053 | 0.0112 | 0.0198 | 0.0298 |
| Buy again + popularity | 0.0233 | 0.0267 | 0.0458 | 0.0561 |
| Ranker (mean of 3 seeds) | 0.0279 | 0.0306 | 0.0548 | 0.0635 |
| Ranker gain over buy again | +19.6% | +14.9% | +19.6% | +13.3% |

The ranking of the models holds, and the ranker's gain is smaller on the test week (14.9% vs. 19.6% on MAP@12; NDCG@12
15.3% vs. 19.3%), as expected when the number of training weeks was chosen on the validation week. All models score higher on
the test week, popularity most of all (0.0053 to 0.0112), so levels are not comparable between weeks. The candidates
contain 16.9% of the purchased articles on the test week (14.6% on validation).

**By segment** (MAP@12 and recall@12, ranker seed 0):

| Segment | Customers | Popularity | Buy again + popularity | Ranker |
|---|---|---|---|---|
| Returning (bought before in the window) | 5,360 | 0.0112 / 0.0293 | 0.0313 / 0.0635 | **0.0376 / 0.0740** |
| No history | 1,612 | 0.0113 / 0.0314 | 0.0113 / 0.0314 | 0.0068 / 0.0255 |
| Age < 25 | 1,872 | 0.0155 / 0.0423 | 0.0290 / 0.0633 | 0.0311 / 0.0717 |
| Age 25-34 | 2,219 | 0.0110 / 0.0273 | 0.0310 / 0.0582 | 0.0340 / 0.0636 |
| Age 35-44 | 719 | 0.0108 / 0.0238 | 0.0230 / 0.0488 | 0.0282 / 0.0499 |
| Age 45-54 | 1,285 | 0.0061 / 0.0201 | 0.0237 / 0.0501 | 0.0290 / 0.0594 |
| Age 55+ | 852 | 0.0103 / 0.0283 | 0.0186 / 0.0497 | 0.0245 / 0.0564 |
| Age unknown | 25 | 0.0200 / 0.0400 | 0.0077 / 0.0533 | 0.0233 / 0.0867 |

- **Returning customers:** the ranker is 20.2% better than "buy again" on MAP@12 (0.0376 vs. 0.0313) and its recall@12
  is 0.0740 vs. 0.0635.
- **Customers with no history are worse off with the ranker:** 0.0068 vs. 0.0113 for plain popularity (about 40% lower),
  and recall@12 0.0255 vs. 0.0314. Their candidates are only popular articles, so the ranker has little to rank on and
  does worse than the popularity order. On the validation week the same comparison went the other way (0.0075 vs. 0.0062),
  so this is week-dependent, but it is a reason to give these customers the popularity list directly. The service
  already does: its table only holds customers who have purchase history, and every other id falls back to popularity.
  This finding is a post-hoc observation from the test week, not a hypothesis I planned to test, so I did not
  change anything because of it and report no adjusted number.
- **Age groups:** the ranker is ahead of "buy again" in every group; the groups are small, so the differences between
  them are noisy, and the 25 customers with unknown age say nothing.
