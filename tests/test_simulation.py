from datetime import date

import numpy as np
import pandas as pd

from hm_recsys.experiments import required_sample_size_mean
from hm_recsys.simulation import (
    calibrate_boundaries,
    cuped_adjust,
    experiment_table,
    null_walk_z,
    peeking_z_paths,
    regression_variance_reduction,
    sequential_outcome,
    simulate_power,
    variance_reduction,
)

EXP = date(2020, 9, 1)


def _tx(rows):
    return pd.DataFrame(rows, columns=["t_dat", "customer_id", "article_id", "price"])


def test_experiment_table_windows_and_no_leakage():
    pre = [
        (date(2020, 7, 10), "a", 1, 2.0),
        (date(2020, 8, 30), "a", 2, 3.0),
        (date(2020, 8, 20), "b", 1, 5.0),
        (date(2020, 6, 1), "z", 1, 9.0),  # before the 56-day pre-period: not in the population
    ]
    exp = [(date(2020, 9, 1), "a", 3, 4.0), (date(2020, 9, 14), "a", 3, 1.0)]
    late = [(date(2020, 9, 15), "b", 1, 50.0)]  # after the 14-day window: ignored
    only_exp = [(date(2020, 9, 2), "c", 1, 7.0)]  # no pre-period purchase: not in the population
    base = experiment_table(_tx(pre + exp), EXP)
    assert list(base["customer_id"]) == ["a", "b"]
    a = base.set_index("customer_id").loc["a"]
    assert a["pre_spend"] == 5.0 and a["pre_orders"] == 2
    assert a["y_rev"] == 5.0 and a["y_buy"] == 1
    assert base.set_index("customer_id").loc["b", "y_buy"] == 0
    # purchases in or after the experiment window can change outcomes, never the covariate
    more = experiment_table(_tx(pre + exp + late + only_exp), EXP)
    assert list(more["customer_id"]) == ["a", "b"]
    pd_cols = ["customer_id", "pre_spend", "pre_orders"]
    pd.testing.assert_frame_equal(base[pd_cols], more[pd_cols])


def test_cuped_variance_reduction_matches_squared_correlation_and_keeps_the_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50_000)
    y = 0.6 * x + 0.8 * rng.normal(size=50_000) + 3.0  # rho = 0.6
    assert abs(variance_reduction(y, x) - 0.36) < 0.01
    adj = cuped_adjust(y, x)
    assert abs(adj.mean() - y.mean()) < 1e-9
    assert adj.var() < 0.7 * y.var()
    assert abs(regression_variance_reduction(y, x.reshape(-1, 1)) - 0.36) < 0.01


def test_simulated_power_false_positives_unbiasedness_and_cuped_gain():
    rng = np.random.default_rng(1)
    x = rng.gamma(2.0, 1.0, 20_000)
    y = 0.8 * x + rng.gamma(2.0, 1.0, 20_000)
    null = simulate_power(y, x, 400, 0.0, sims=600, seed=2)
    assert 0.02 < null["rejection_rate"] < 0.09
    plain = simulate_power(y, x, 400, 0.10, sims=400, seed=3)
    cuped = simulate_power(y, x, 400, 0.10, sims=400, seed=3, cuped=True)
    assert abs(plain["mean_relative_estimate"] - 0.10) < 0.02
    assert abs(cuped["mean_relative_estimate"] - 0.10) < 0.02  # CUPED does not bias the estimate
    assert cuped["sd_relative_estimate"] < plain["sd_relative_estimate"]
    assert cuped["rejection_rate"] > plain["rejection_rate"]


def test_pocock_and_obf_boundaries_match_known_values():
    pocock1, obf1 = calibrate_boundaries(1, sims=200_000)
    assert abs(pocock1 - 1.96) < 0.03 and abs(obf1 - 1.96) < 0.03
    pocock5, obf5 = calibrate_boundaries(5, sims=200_000)
    assert abs(pocock5 - 2.413) < 0.04  # textbook Pocock constant for 5 looks
    assert 1.96 < obf5 < 2.2  # O'Brien-Fleming-type constant is close to the fixed-sample value


def test_peeking_inflates_false_positives_and_boundaries_repair_it():
    looks = 10
    z = null_walk_z(looks, 40_000, seed=5)
    naive = sequential_outcome(z, np.full(looks, 1.96))
    assert 0.17 < naive["rejection_rate"] < 0.22  # about 19% with 10 looks at nominal 5%
    pocock, obf = calibrate_boundaries(looks, sims=200_000, seed=6)
    fixed = sequential_outcome(z, np.full(looks, pocock))
    spend = sequential_outcome(z, obf / np.sqrt(np.arange(1, looks + 1) / looks))
    assert 0.04 < fixed["rejection_rate"] < 0.06
    assert 0.04 < spend["rejection_rate"] < 0.06


def test_peeking_paths_on_data_are_standard_normal_under_the_null():
    rng = np.random.default_rng(7)
    y = rng.exponential(1.0, 5000) * (rng.random(5000) < 0.3)
    z = peeking_z_paths(y, 400, looks=4, rel_effect=0.0, sims=800, seed=8)
    assert z.shape == (800, 4)
    assert abs(z[:, -1].mean()) < 0.15 and 0.85 < z[:, -1].std() < 1.15
    with_effect = peeking_z_paths(y, 400, looks=4, rel_effect=0.5, sims=300, seed=8)
    assert with_effect[:, -1].mean() > 1.0


def test_required_sample_size_for_a_mean():
    # sd 10, detect a change of 1: n = 2 * ((1.96 + 0.8416) * 10)^2 = 1570
    assert abs(required_sample_size_mean(10, 1) - 1570) <= 1
    assert required_sample_size_mean(10, 0.5) > 3.9 * required_sample_size_mean(10, 1)
