"""A/B test statistics: sample ratio mismatch, proportion tests, bootstrap, power.

Convention: arm A is the control, arm B the treatment, and every difference is B minus A.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy import stats


def srm_test(n_a: int, n_b: int, expected_share_a: float = 0.5) -> dict:
    """Sample ratio mismatch: chi-square goodness-of-fit of the observed arm sizes against the
    intended split. A small p-value means assignment (or logging) did not follow the design."""
    total = n_a + n_b
    expected = [total * expected_share_a, total * (1 - expected_share_a)]
    chi2 = sum((o - e) ** 2 / e for o, e in zip([n_a, n_b], expected, strict=True))
    return {"chi2": chi2, "p_value": float(stats.chi2.sf(chi2, df=1)), "expected": expected}


def two_proportion_test(x_a: int, n_a: int, x_b: int, n_b: int, alpha: float = 0.05) -> dict:
    """Difference in proportions (B - A).

    The p-value comes from the pooled z-test (variance under H0: both arms share one rate). The
    confidence interval uses the unpooled standard error, since it describes the difference
    actually observed, not the null."""
    p_a, p_b = x_a / n_a, x_b / n_b
    diff = p_b - p_a
    pooled = (x_a + x_b) / (n_a + n_b)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = diff / se_pooled
    se = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    crit = stats.norm.ppf(1 - alpha / 2)
    return {
        "p_a": p_a,
        "p_b": p_b,
        "diff": diff,
        "relative_lift": diff / p_a,
        "z": z,
        "p_value": float(2 * stats.norm.sf(abs(z))),
        "se": se,
        "ci": (diff - crit * se, diff + crit * se),
    }


def bootstrap_proportion_diff(
    x_a: int, n_a: int, x_b: int, n_b: int, n_boot: int = 10_000, seed: int = 0, alpha=0.05
) -> dict:
    """Percentile bootstrap CI for the difference in proportions. Resampling users with
    replacement from a 0/1 outcome gives a Binomial(n, p_hat) count, so that is drawn directly."""
    rng = np.random.default_rng(seed)
    p_a = rng.binomial(n_a, x_a / n_a, n_boot) / n_a
    p_b = rng.binomial(n_b, x_b / n_b, n_boot) / n_b
    d = p_b - p_a
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"ci": (float(lo), float(hi)), "n_boot": n_boot}


def bootstrap_diff(
    a: np.ndarray,
    b: np.ndarray,
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap CI for stat(b) - stat(a), resampling each arm with replacement."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = stat(rng.choice(b, len(b))) - stat(rng.choice(a, len(a)))
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "diff": float(stat(b) - stat(a)),
        "ci": (float(lo), float(hi)),
        "share_above_zero": float((diffs > 0).mean()),
        "n_boot": n_boot,
    }


def required_sample_size(p_base: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8):
    """Users per arm to detect an absolute change `mde_abs` from baseline rate `p_base` with a
    two-sided two-proportion z-test."""
    p2 = p_base + mde_abs
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    var = p_base * (1 - p_base) + p2 * (1 - p2)
    return math.ceil((z_a + z_b) ** 2 * var / mde_abs**2)


def required_sample_size_mean(
    sd: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Users per arm to detect an absolute change `mde_abs` in a mean whose standard deviation is
    `sd` (same in both arms), with a two-sided z/t-test."""
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return math.ceil(2 * ((z_a + z_b) * sd / mde_abs) ** 2)


def minimum_detectable_effect(
    p_base: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Smallest absolute change detectable with `n_per_arm` users per arm (found by bisection,
    since the variance depends on the effect)."""
    lo, hi = 1e-6, 1 - p_base - 1e-6
    for _ in range(60):
        mid = (lo + hi) / 2
        if required_sample_size(p_base, mid, alpha, power) > n_per_arm:
            lo = mid
        else:
            hi = mid
    return hi
