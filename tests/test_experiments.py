import math

import numpy as np

from hm_recsys.experiments import (
    bootstrap_diff,
    bootstrap_proportion_diff,
    minimum_detectable_effect,
    required_sample_size,
    srm_test,
    two_proportion_test,
)


def test_srm_balanced_and_unbalanced():
    even = srm_test(100, 100)
    assert even["chi2"] == 0 and abs(even["p_value"] - 1.0) < 1e-12
    uneven = srm_test(120, 80)  # chi2 = 2 * 20^2 / 100 = 8
    assert abs(uneven["chi2"] - 8.0) < 1e-12
    assert abs(uneven["p_value"] - 0.004678) < 1e-5
    assert srm_test(60, 40, expected_share_a=0.6)["chi2"] == 0


def test_two_proportion_test_matches_hand_calculation():
    r = two_proportion_test(100, 1000, 130, 1000)
    assert abs(r["diff"] - 0.03) < 1e-12
    assert abs(r["relative_lift"] - 0.3) < 1e-12
    pooled = 230 / 2000
    z = 0.03 / math.sqrt(pooled * (1 - pooled) * 2 / 1000)
    assert abs(r["z"] - z) < 1e-12
    assert abs(z - 2.1027) < 1e-3  # normal table: two-sided p at z = 2.10 is about 0.036
    assert abs(r["p_value"] - 0.0355) < 5e-4
    se = math.sqrt(0.1 * 0.9 / 1000 + 0.13 * 0.87 / 1000)
    assert abs(r["ci"][0] - (0.03 - 1.959964 * se)) < 1e-6
    assert r["ci"][0] < r["diff"] < r["ci"][1]


def test_identical_arms_give_p_one_and_symmetric_ci():
    r = two_proportion_test(50, 500, 50, 500)
    assert r["diff"] == 0 and abs(r["p_value"] - 1.0) < 1e-12
    assert abs(r["ci"][0] + r["ci"][1]) < 1e-12


def test_bootstrap_proportion_ci_agrees_with_analytic():
    boot = bootstrap_proportion_diff(2000, 10_000, 2150, 10_000, n_boot=20_000, seed=1)
    analytic = two_proportion_test(2000, 10_000, 2150, 10_000)["ci"]
    assert abs(boot["ci"][0] - analytic[0]) < 5e-4
    assert abs(boot["ci"][1] - analytic[1]) < 5e-4


def test_bootstrap_diff_of_means_covers_the_true_difference():
    rng = np.random.default_rng(0)
    a, b = rng.normal(10, 2, 2000), rng.normal(10.5, 2, 2000)
    r = bootstrap_diff(a, b, np.mean, n_boot=500, seed=2)
    assert r["ci"][0] < 0.5 < r["ci"][1]
    assert abs(r["diff"] - (b.mean() - a.mean())) < 1e-12


def test_sample_size_and_mde_are_consistent():
    # 10% -> 12% at alpha 0.05 and power 0.8 needs about 3,800 per arm (textbook value)
    n = required_sample_size(0.10, 0.02)
    assert 3700 < n < 3950
    assert abs(minimum_detectable_effect(0.10, n) - 0.02) < 1e-3
    assert required_sample_size(0.10, 0.01) > 3 * n  # half the effect needs about 4x the users


def test_false_positive_rate_in_an_aa_simulation_is_close_to_alpha():
    rng = np.random.default_rng(3)
    n, p, sims = 2000, 0.2, 1500
    xs = rng.binomial(n, p, (sims, 2))
    rejected = sum(two_proportion_test(int(a), n, int(b), n)["p_value"] < 0.05 for a, b in xs)
    assert 0.03 < rejected / sims < 0.07
