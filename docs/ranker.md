# Ranker

Part of [hm-recsys](../README.md). Every number here comes from the JSON files in `results/`.

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
tuned. Same 10% customer sample and validation week as in the baselines and candidate sections. Produced by
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
so 0.0279 is a little optimistic. The test week was not used at this stage; see the [final evaluation](final-evaluation.md) for
the result there.

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
