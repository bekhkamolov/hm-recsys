"""Experiment simulations calibrated on H&M purchase behaviour.

The population is real customers with their real outcomes; what is simulated is the random
assignment, the treatment effect, and (for peeking) the order in which users arrive.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05


def experiment_table(
    df: pd.DataFrame, exp_start: date, exp_days: int = 14, pre_days: int = 56
) -> pd.DataFrame:
    """One row per customer who bought in the `pre_days` before `exp_start`.

    pre_spend: spend in the pre-period (the CUPED covariate, known before assignment).
    y_buy / y_rev: bought anything / total spend in the `exp_days` from `exp_start`.
    Prices in this dataset are scaled, so spend is in price units, not currency.
    """
    t = pd.to_datetime(df["t_dat"])
    start = pd.Timestamp(exp_start)
    pre = df[(t >= start - pd.Timedelta(days=pre_days)) & (t < start)]
    exp = df[(t >= start) & (t < start + pd.Timedelta(days=exp_days))]
    table = pre.groupby("customer_id")["price"].agg(pre_spend="sum", pre_orders="size")
    spend = exp.groupby("customer_id")["price"].sum().rename("y_rev")
    table = table.join(spend).fillna({"y_rev": 0.0})
    table["y_buy"] = (table["y_rev"] > 0).astype(int)
    return table.reset_index()


def cuped_theta(y: np.ndarray, x: np.ndarray) -> float:
    """Regression coefficient of the outcome on the pre-period covariate."""
    return float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))


def cuped_adjust(y: np.ndarray, x: np.ndarray, theta: float | None = None) -> np.ndarray:
    """Y - theta * (X - mean(X)). The mean is unchanged, the variance shrinks by about rho^2.
    theta is estimated on both arms pooled so the adjustment cannot depend on the assignment."""
    theta = cuped_theta(y, x) if theta is None else theta
    return y - theta * (x - x.mean())


def variance_reduction(y: np.ndarray, x: np.ndarray) -> float:
    """Share of the outcome variance removed by CUPED (equals the squared correlation)."""
    return float(1 - np.var(cuped_adjust(y, x), ddof=1) / np.var(y, ddof=1))


def regression_variance_reduction(y: np.ndarray, covariates: np.ndarray) -> float:
    """Share of the variance of y explained by a linear regression on one or more pre-period
    covariates (the multi-covariate generalisation of CUPED; equals R^2)."""
    X = np.column_stack([np.ones(len(y)), covariates])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(1 - resid.var(ddof=1) / y.var(ddof=1))


def _welch_z(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float(diff), float(diff / se)


def simulate_power(
    y: np.ndarray,
    x: np.ndarray,
    n_per_arm: int,
    rel_effect: float,
    sims: int = 1000,
    cuped: bool = False,
    alpha: float = ALPHA,
    seed: int = 0,
) -> dict:
    """Share of simulated experiments that reject H0, for a treatment that multiplies each treated
    customer's outcome by (1 + rel_effect). Customers are drawn with replacement from the real
    population (y, x), then split into two arms. rel_effect = 0 gives the false-positive rate.
    Also returns the mean estimated effect (relative to the control mean) and its spread."""
    rng = np.random.default_rng(seed)
    crit = stats.norm.ppf(1 - alpha / 2)
    rejected, estimates = 0, []
    for _ in range(sims):
        idx = rng.integers(0, len(y), 2 * n_per_arm)
        yy, xx = y[idx].copy(), x[idx]
        yy[n_per_arm:] *= 1 + rel_effect
        if cuped:
            yy = cuped_adjust(yy, xx)
        diff, z = _welch_z(yy[:n_per_arm], yy[n_per_arm:])
        rejected += abs(z) > crit
        estimates.append(diff / y[idx[:n_per_arm]].mean())
    return {
        "rejection_rate": rejected / sims,
        "mean_relative_estimate": float(np.mean(estimates)),
        "sd_relative_estimate": float(np.std(estimates, ddof=1)),
        "sims": sims,
    }


def null_walk_z(looks: int, sims: int, seed: int = 0) -> np.ndarray:
    """z statistics at `looks` equally spaced looks under H0 (a Gaussian random walk)."""
    rng = np.random.default_rng(seed)
    steps = rng.standard_normal((sims, looks))
    return np.cumsum(steps, axis=1) / np.sqrt(np.arange(1, looks + 1))


def calibrate_boundaries(looks: int, alpha: float = ALPHA, sims: int = 400_000, seed: int = 0):
    """Critical values for a group-sequential test with `looks` equally spaced looks, found by
    simulating the null so that the chance of ever crossing is `alpha`.

    Pocock: the same z at every look. O'Brien-Fleming style: z_k = c / sqrt(k / looks), strict
    early, close to the ordinary critical value at the end. Returns (pocock_z, obf_c)."""
    z = np.abs(null_walk_z(looks, sims, seed))
    t = np.arange(1, looks + 1) / looks
    pocock = float(np.quantile(z.max(axis=1), 1 - alpha))
    obf = float(np.quantile((z * np.sqrt(t)).max(axis=1), 1 - alpha))
    return pocock, obf


def peeking_z_paths(
    y: np.ndarray, n_per_arm: int, looks: int, rel_effect: float, sims: int, seed: int = 0
) -> np.ndarray:
    """Welch z statistic at `looks` equally spaced sample sizes for `sims` simulated experiments
    on real outcomes. Users arrive in random order; the first `n_per_arm * k / looks` per arm
    are analysed at look k. Returns an array of shape (sims, looks)."""
    rng = np.random.default_rng(seed)
    sizes = (n_per_arm * np.arange(1, looks + 1) // looks).astype(int)
    out = np.empty((sims, looks))
    batch = 50
    for s in range(0, sims, batch):
        b = min(batch, sims - s)
        a = y[rng.integers(0, len(y), (b, n_per_arm))]
        c = y[rng.integers(0, len(y), (b, n_per_arm))] * (1 + rel_effect)
        ca, cqa = np.cumsum(a, axis=1), np.cumsum(a**2, axis=1)
        cc, cqc = np.cumsum(c, axis=1), np.cumsum(c**2, axis=1)
        for k, n in enumerate(sizes):
            ma, mc = ca[:, n - 1] / n, cc[:, n - 1] / n
            va = (cqa[:, n - 1] - n * ma**2) / (n - 1)
            vc = (cqc[:, n - 1] - n * mc**2) / (n - 1)
            out[s : s + b, k] = (mc - ma) / np.sqrt(va / n + vc / n)
    return out


def sequential_outcome(z: np.ndarray, bounds: np.ndarray) -> dict:
    """Apply per-look critical values `bounds` to z paths: share of experiments that ever reject
    and the mean share of the planned sample used (stopping at the first rejection)."""
    crossed = np.abs(z) > bounds[None, :]
    ever = crossed.any(axis=1)
    first = np.where(ever, crossed.argmax(axis=1) + 1, z.shape[1])
    return {
        "rejection_rate": float(ever.mean()),
        "mean_share_of_planned_sample_used": float((first / z.shape[1]).mean()),
    }
