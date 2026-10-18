"""Analyse the public Cookie Cats A/B test (gate moved from level 30 to level 40).

Control (A) = gate_30, treatment (B) = gate_40. All differences are B - A. Writes
results/ab_cookie_cats.json. Run scripts/download_ab_data.py first.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from hm_recsys.experiments import (
    bootstrap_diff,
    bootstrap_proportion_diff,
    minimum_detectable_effect,
    required_sample_size,
    srm_test,
    two_proportion_test,
)

ROOT = Path(__file__).resolve().parents[1]
PRACTICAL_PP = 0.005  # assumption, not from the data: 0.5 percentage points of retention


def verdict(ci: tuple[float, float], threshold: float) -> str:
    """Where the CI sits relative to zero and to +-threshold (smallest change worth acting on)."""
    lo, hi = ci
    if lo > 0 or hi < 0:
        if lo > threshold or hi < -threshold:
            return (
                "statistically significant and the whole interval is beyond the practical threshold"
            )
        if abs(lo) < threshold and abs(hi) < threshold:
            return (
                "statistically significant but the whole interval is below the practical threshold"
            )
        return "statistically significant; the interval spans small and practically relevant sizes"
    if lo > -threshold and hi < threshold:
        return "not significant, and the interval rules out practically relevant effects"
    return "not significant, and the interval does not rule out practically relevant effects"


def main() -> None:
    d = pd.read_csv(ROOT / "data" / "raw" / "cookie_cats" / "cookie_cats.csv")
    a = d[d["version"] == "gate_30"]
    b = d[d["version"] == "gate_40"]
    n_a, n_b = len(a), len(b)
    out: dict = {"n_users": len(d), "n_gate_30": n_a, "n_gate_40": n_b}

    # data checks
    out["checks"] = {
        "duplicate_userids": int(d["userid"].duplicated().sum()),
        "missing_values": int(d.isna().sum().sum()),
        "users_with_zero_rounds_share": float((d["sum_gamerounds"] == 0).mean()),
        "retention_1_among_zero_rounds": float(
            d.loc[d["sum_gamerounds"] == 0, "retention_1"].mean()
        ),
        "top5_rounds_by_arm": {
            v: sorted(g["sum_gamerounds"].nlargest(5).tolist(), reverse=True)
            for v, g in d.groupby("version")
        },
    }

    # 1. sample ratio mismatch (assuming the intended split was 50/50)
    out["srm"] = srm_test(n_a, n_b)

    # 2. retention: difference in proportions, analytic and bootstrap
    out["retention"] = {}
    for metric in ("retention_1", "retention_7"):
        x_a, x_b = int(a[metric].sum()), int(b[metric].sum())
        t = two_proportion_test(x_a, n_a, x_b, n_b)
        t_bonf = two_proportion_test(x_a, n_a, x_b, n_b, alpha=0.025)
        boot = bootstrap_proportion_diff(x_a, n_a, x_b, n_b)
        t["ci"] = list(t["ci"])
        out["retention"][metric] = {
            **t,
            "x_a": x_a,
            "x_b": x_b,
            "ci_bonferroni_2_metrics": list(t_bonf["ci"]),
            "bootstrap_ci": list(boot["ci"]),
            "verdict_vs_0.5pp": verdict(tuple(t["ci"]), PRACTICAL_PP),
        }

    # 3. game rounds: heavy-tailed, so compare tests and robustness to the outlier
    ra, rb = a["sum_gamerounds"].to_numpy(), b["sum_gamerounds"].to_numpy()
    cap = float(np.percentile(d["sum_gamerounds"], 99.9))
    rounds = {
        "mean": {"gate_30": float(ra.mean()), "gate_40": float(rb.mean())},
        "median": {"gate_30": float(np.median(ra)), "gate_40": float(np.median(rb))},
        "welch_t_p": float(stats.ttest_ind(rb, ra, equal_var=False).pvalue),
        "mann_whitney_p": float(stats.mannwhitneyu(rb, ra, alternative="two-sided").pvalue),
        "bootstrap_mean_diff": bootstrap_diff(ra, rb, np.mean),
        "bootstrap_median_diff": bootstrap_diff(ra, rb, np.median),
        "winsorize_at_99.9_pct": cap,
        "bootstrap_mean_diff_winsorized": bootstrap_diff(
            np.minimum(ra, cap), np.minimum(rb, cap), np.mean
        ),
        "mean_diff_without_largest_value": float(rb.mean() - np.delete(ra, ra.argmax()).mean()),
    }
    for k in ("bootstrap_mean_diff", "bootstrap_median_diff", "bootstrap_mean_diff_winsorized"):
        rounds[k]["ci"] = list(rounds[k]["ci"])
    out["game_rounds"] = rounds

    # 4. A/A check of the pipeline: shuffle arm labels, how often is p < 0.05?
    rng = np.random.default_rng(0)
    sims = 20_000
    aa = {}
    for metric in ("retention_1", "retention_7"):
        total = int(d[metric].sum())
        x_a = rng.hypergeometric(total, len(d) - total, n_a, sims)  # same as shuffling labels
        x_b = total - x_a
        pooled = total / len(d)
        se = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
        z = (x_b / n_b - x_a / n_a) / se
        aa[metric] = float((2 * stats.norm.sf(np.abs(z)) < 0.05).mean())
    out["aa_false_positive_rate_at_0.05"] = {"simulations": sims, **aa}

    # 5. power: what could this experiment detect, and what would a smaller effect need?
    per_arm = min(n_a, n_b)
    base1, base7 = float(a["retention_1"].mean()), float(a["retention_7"].mean())
    out["power"] = {
        "alpha": 0.05,
        "power": 0.8,
        "min_detectable_effect_retention_1": minimum_detectable_effect(base1, per_arm),
        "min_detectable_effect_retention_7": minimum_detectable_effect(base7, per_arm),
        "users_per_arm_for_0.5pp_on_retention_7": required_sample_size(base7, 0.005),
        "users_per_arm_for_1.0pp_on_retention_7": required_sample_size(base7, 0.01),
        "users_per_arm_for_0.5pp_on_retention_1": required_sample_size(base1, 0.005),
    }

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "ab_cookie_cats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
