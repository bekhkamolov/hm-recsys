"""Experiment design simulations calibrated on H&M purchase behaviour (Day 7).

Population: customers who bought in the 56 days before 2020-08-12 (the covariate window).
Outcome window: the 14 days from 2020-08-12. Both lie before the validation and test weeks.
Writes results/experiment_sim.json.
"""

import json
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from hm_recsys.experiments import (
    minimum_detectable_effect,
    required_sample_size,
    required_sample_size_mean,
)
from hm_recsys.simulation import (
    calibrate_boundaries,
    experiment_table,
    peeking_z_paths,
    regression_variance_reduction,
    sequential_outcome,
    simulate_power,
    variance_reduction,
)

ROOT = Path(__file__).resolve().parents[1]
EXP_START = date(2020, 8, 12)
EXP_DAYS, PRE_DAYS = 14, 56
Z_A, Z_B = stats.norm.ppf(0.975), stats.norm.ppf(0.8)
UPLIFTS = [0.01, 0.02, 0.05, 0.10]
LOOKS = [1, 2, 5, 10, 14, 20]
PEEK_SIMS = 3000


def main() -> None:
    processed = ROOT / "data" / "processed"
    info = json.loads((processed / "sample.json").read_text())
    df = pd.read_parquet(processed / "transactions.parquet")
    table = experiment_table(df, EXP_START, EXP_DAYS, PRE_DAYS)
    y, x = table["y_rev"].to_numpy(), table["pre_spend"].to_numpy()
    buy = table["y_buy"].to_numpy()
    n_pop, p, mean, sd = len(table), buy.mean(), y.mean(), y.std(ddof=1)
    vr_rev = variance_reduction(y, x)
    out: dict = {
        "sample": info,
        "experiment_window": [str(EXP_START), str(EXP_START + timedelta(days=EXP_DAYS))],
        "pre_period": [str(EXP_START - timedelta(days=PRE_DAYS)), str(EXP_START)],
        "population_in_sample": n_pop,
    }

    # full-data population size, for the feasibility question
    lo, hi = EXP_START - timedelta(days=PRE_DAYS), EXP_START
    csv = (ROOT / "data" / "raw" / "transactions_train.csv").as_posix()
    full = (
        duckdb.connect()
        .execute(
            f"SELECT count(DISTINCT customer_id) FROM read_csv('{csv}', header=true) "
            f"WHERE t_dat >= DATE '{lo}' AND t_dat < DATE '{hi}'"
        )
        .fetchone()[0]
    )
    out["eligible_customers_full_data"] = int(full)

    # 1. what the real population looks like
    out["population"] = {
        "purchase_rate_14d": float(p),
        "revenue_per_user_mean": float(mean),
        "revenue_per_user_sd": float(sd),
        "revenue_cv": float(sd / mean),
        "corr_revenue_pre_spend": float(np.corrcoef(y, x)[0, 1]),
        "corr_purchase_pre_spend": float(np.corrcoef(buy, x)[0, 1]),
        "cuped_variance_reduction_revenue": vr_rev,
        "cuped_variance_reduction_purchase": variance_reduction(buy.astype(float), x),
        "two_covariate_variance_reduction_revenue": regression_variance_reduction(
            y, table[["pre_spend", "pre_orders"]].to_numpy()
        ),
    }

    # 2. analytic sample sizes (users per arm, 80% power, alpha 0.05)
    sizes = {}
    for u in UPLIFTS:
        n_rev = required_sample_size_mean(sd, mean * u)
        sizes[f"{u:.0%}"] = {
            "purchase_rate": required_sample_size(p, p * u),
            "revenue_per_user": n_rev,
            "revenue_per_user_cuped": required_sample_size_mean(sd * np.sqrt(1 - vr_rev), mean * u),
        }
    out["users_per_arm_for_relative_uplift"] = sizes

    # smallest relative uplift detectable if the whole eligible base is split in two
    def mde(n_arm: int, cv: float) -> float:
        return float((Z_A + Z_B) * cv * np.sqrt(2 / n_arm))

    cv = sd / mean
    cv_c = cv * np.sqrt(1 - vr_rev)
    out["detectable_relative_uplift_revenue"] = {
        "in_sample_population": {"naive": mde(n_pop // 2, cv), "cuped": mde(n_pop // 2, cv_c)},
        "full_data_population": {
            "naive": mde(full // 2, cv),
            "cuped": mde(full // 2, cv_c),
        },
    }

    out["detectable_relative_uplift_purchase_rate"] = {
        "in_sample_population": minimum_detectable_effect(p, n_pop // 2) / p,
        "full_data_population": minimum_detectable_effect(p, full // 2) / p,
    }

    # 3. simulation check of the formulas and of CUPED (draws from the real customers)
    n10 = sizes["10%"]["revenue_per_user"]
    n5 = sizes["5%"]["revenue_per_user"]
    sim = {"n_per_arm_10pct": n10, "n_per_arm_5pct": n5}
    for label, n, eff, sims in (
        ("aa_at_n10", n10, 0.0, 1000),
        ("effect_10pct_at_n10", n10, 0.10, 1000),
        ("effect_5pct_at_n5", n5, 0.05, 500),
    ):
        sim[label] = {
            "naive": simulate_power(y, x, n, eff, sims, cuped=False, seed=1),
            "cuped": simulate_power(y, x, n, eff, sims, cuped=True, seed=1),
        }
    out["simulation_power"] = sim

    # 4. peeking: A/A tests on real revenue, stop at the first p < 0.05 across the looks
    peek = {"planned_users_per_arm": n10, "sims": PEEK_SIMS, "naive_false_positive_rate": {}}
    paths = {}
    for k in LOOKS:
        paths[k] = peeking_z_paths(y, n10, k, 0.0, PEEK_SIMS, seed=10 + k)
        peek["naive_false_positive_rate"][str(k)] = sequential_outcome(paths[k], np.full(k, Z_A))[
            "rejection_rate"
        ]
        print(
            f"looks={k}: false positive rate {peek['naive_false_positive_rate'][str(k)]:.3f}",
            flush=True,
        )

    k = 14  # one look per day over the 14-day experiment
    pocock, obf_c = calibrate_boundaries(k)
    t = np.arange(1, k + 1) / k
    rules = {
        "fixed horizon (look once at the end)": np.r_[np.full(k - 1, np.inf), Z_A],
        "peeking daily at 1.96": np.full(k, Z_A),
        "Bonferroni (alpha/14 per look)": np.full(k, stats.norm.isf(0.05 / (2 * k))),
        "Pocock (constant boundary)": np.full(k, pocock),
        "O'Brien-Fleming style": obf_c / np.sqrt(t),
    }
    alt = peeking_z_paths(y, n10, k, 0.10, PEEK_SIMS, seed=99)
    peek["boundaries_14_looks"] = {"pocock_z": pocock, "obf_c": obf_c}
    peek["rules_14_looks"] = {
        name: {
            "null": sequential_outcome(paths[k], b),
            "effect_10pct": sequential_outcome(alt, b),
        }
        for name, b in rules.items()
    }
    out["peeking"] = peek

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "experiment_sim.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
