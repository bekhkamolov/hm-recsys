import numpy as np
import pandas as pd
from scipy import stats

from hm_recsys.monitoring import drift_report, noise_floor, psi, psi_categorical, status


def test_identical_samples_have_zero_psi_and_resamples_have_almost_none():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50_000)
    assert psi(x, x) == 0.0
    assert psi(x, rng.normal(size=50_000)) < 0.005


def test_psi_of_a_shifted_normal_matches_the_formula():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=400_000)
    cur = rng.normal(loc=1.0, size=400_000)
    # independent calculation: decile edges of N(0,1), bin shares of N(1,1) from the cdf
    edges = stats.norm.ppf(np.linspace(0, 1, 11)[1:-1])
    p = np.full(10, 0.1)
    cdf = np.r_[0, stats.norm.cdf(edges - 1.0), 1]
    q = np.diff(cdf)
    expected = float(np.sum((q - p) * np.log(q / p)))
    assert abs(psi(ref, cur) - expected) < 0.01
    assert expected > 0.25  # a one-standard-deviation shift is a major change


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(2)
    ref = rng.normal(size=20_000)
    values = [psi(ref, rng.normal(loc=m, size=20_000)) for m in (0.0, 0.2, 0.5, 1.0)]
    assert values == sorted(values) and values[0] < 0.01 < values[-1]


def test_missing_values_are_their_own_bin():
    rng = np.random.default_rng(3)
    ref = rng.normal(size=10_000)
    cur = ref.copy()
    cur[:3000] = np.nan  # 30% of the current sample is suddenly missing
    assert psi(ref, cur) > 0.25
    assert psi(np.r_[ref, [np.nan] * 100], np.r_[ref, [np.nan] * 100]) == 0.0


def test_constant_and_heavily_tied_features():
    assert psi(np.zeros(1000), np.zeros(1000)) == 0.0
    assert psi(np.zeros(1000), np.ones(1000)) > 0.25
    tied = np.r_[np.zeros(900), np.arange(100)]  # most values identical: fewer bins, no crash
    assert psi(tied, tied) == 0.0


def test_categorical_psi():
    ref = np.repeat([1, 2, 3], [500, 300, 200])
    assert psi_categorical(ref, ref) == 0.0
    same_mix = np.repeat([1, 2, 3], [50, 30, 20])
    assert psi_categorical(ref, same_mix) < 0.001
    new_category = np.repeat([1, 2, 3, 99], [400, 250, 150, 200])
    assert psi_categorical(ref, new_category) > 0.25


def test_status_thresholds():
    assert [status(v) for v in (0.0, 0.0999, 0.1, 0.2499, 0.25, 3.0)] == [
        "stable",
        "stable",
        "moderate",
        "moderate",
        "major",
        "major",
    ]


def test_drift_report_ranks_the_drifted_feature_first_and_noise_floor_is_small():
    rng = np.random.default_rng(4)
    n = 20_000
    ref = pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(size=n), "cat": rng.integers(0, 4, n)}
    )
    cur = pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(loc=0.8, size=n), "cat": rng.integers(0, 4, n)}
    )
    report = drift_report(ref, cur, ["a", "b", "cat"], categorical=["cat"])
    assert list(report["feature"])[0] == "b" and report.iloc[0]["status"] == "major"
    assert set(report[report["feature"] != "b"]["status"]) == {"stable"}
    floor = noise_floor(ref, ["a", "b", "cat"], ["cat"])
    assert floor["psi"].max() < 0.01


def test_noise_floor_by_group_keeps_each_group_on_one_side():
    rng = np.random.default_rng(5)
    customers = np.repeat(np.arange(2000), 10)  # 10 rows per customer
    level = np.repeat(rng.normal(size=2000), 10)  # a customer-level trait shared by their rows
    frame = pd.DataFrame({"x": level + 0.01 * rng.normal(size=len(customers))})
    by_row = noise_floor(frame, ["x"])["psi"].iloc[0]
    by_customer = noise_floor(frame, ["x"], groups=customers)["psi"].iloc[0]
    # splitting rows lets every customer appear in both halves, which hides the sampling noise
    assert by_customer > by_row
