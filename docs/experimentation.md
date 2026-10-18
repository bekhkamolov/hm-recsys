# Experimentation

Part of [hm-recsys](../README.md). Every number here comes from the JSON files in `results/`.

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
