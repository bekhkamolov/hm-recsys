"""Drift monitoring: population stability index (PSI) for features and for model scores.

PSI compares how a variable is distributed in a reference sample (for example the training data)
and in a current sample (for example this week's candidates). Bins come from the reference
quantiles, missing values get their own bin, and

    PSI = sum over bins of (current_share - reference_share) * ln(current_share / reference_share)

A common rule of thumb reads below 0.1 as stable, 0.1 to 0.25 as a moderate shift and above 0.25 as
a major one. Those cut-offs are convention, not something validated on this data.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

EPS = 1e-4  # floor for bin shares, so an empty bin does not give an infinite PSI
MODERATE, MAJOR = 0.1, 0.25


def status(value: float) -> str:
    """'stable', 'moderate' or 'major' by the usual PSI cut-offs."""
    return "stable" if value < MODERATE else "moderate" if value < MAJOR else "major"


def _psi_from_counts(ref_counts: np.ndarray, cur_counts: np.ndarray) -> float:
    p = np.maximum(ref_counts / ref_counts.sum(), EPS)
    q = np.maximum(cur_counts / cur_counts.sum(), EPS)
    return float(np.sum((q - p) * np.log(q / p)))


def psi(reference, current, bins: int = 10) -> float:
    """PSI of a numeric variable. NaN is its own bin. A variable with at most `bins` distinct
    reference values (a constant, a count) is binned by exact value; otherwise bin edges are
    reference quantiles."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref_ok = ref[~np.isnan(ref)]
    if len(np.unique(ref_ok)) <= bins:
        return psi_categorical(ref, cur, max_categories=bins)
    interior = np.linspace(0, 1, bins + 1)[1:-1]
    edges = np.unique(np.quantile(ref_ok, interior))

    def counts(x: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(edges, x, side="left")  # bins are (edge_i-1, edge_i]
        idx = np.where(np.isnan(x), len(edges) + 1, idx)  # missing values: last bin
        return np.bincount(idx, minlength=len(edges) + 2).astype(float)

    return _psi_from_counts(counts(ref), counts(cur))


def psi_categorical(reference, current, max_categories: int = 30) -> float:
    """PSI of a categorical variable (category codes). The `max_categories` most common reference
    values keep their own bin, everything else and missing values share two further bins."""
    ref = pd.Series(reference)
    cur = pd.Series(current)
    keep = list(ref.value_counts().index[:max_categories])

    def counts(s: pd.Series) -> np.ndarray:
        in_keep = s.isin(keep)
        by_value = s[in_keep].value_counts().reindex(keep, fill_value=0).to_numpy(dtype=float)
        other = float((~in_keep & s.notna()).sum())
        missing = float(s.isna().sum())
        return np.append(by_value, [other, missing])

    return _psi_from_counts(counts(ref), counts(cur))


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric: Sequence[str],
    categorical: Sequence[str] = (),
    bins: int = 10,
) -> pd.DataFrame:
    """PSI per feature (largest first) with its status."""
    rows = [(f, psi(reference[f], current[f], bins)) for f in numeric if f not in categorical]
    rows += [(f, psi_categorical(reference[f], current[f])) for f in categorical]
    out = pd.DataFrame(rows, columns=["feature", "psi"]).sort_values("psi", ascending=False)
    out["status"] = out["psi"].map(status)
    return out.reset_index(drop=True)


def noise_floor(
    frame: pd.DataFrame,
    numeric: Sequence[str],
    categorical: Sequence[str] = (),
    seed: int = 0,
    groups: Sequence | None = None,
) -> pd.DataFrame:
    """PSI between two random halves of one sample: how large PSI is from sampling noise alone, for
    judging what a real PSI means at this sample size. Pass `groups` (one id per row, for example
    the customer) to split by group instead of by row: rows of the same customer are alike, so a
    row-level split makes the halves look more similar than two different samples would be."""
    rng = np.random.default_rng(seed)
    if groups is None:
        half = rng.random(len(frame)) < 0.5
    else:
        ids = pd.Series(np.asarray(groups))
        unique = ids.unique()
        chosen = set(unique[rng.random(len(unique)) < 0.5])
        half = ids.isin(chosen).to_numpy()
    return drift_report(frame[half], frame[~half], numeric, categorical)
