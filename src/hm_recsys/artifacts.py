"""Build the serving artifacts: a trained ranker plus precomputed candidate rows with features.

Training side only (needs the `train` extra). The service in `api.py` reads what this writes and
imports none of the training code.

Artifacts in `out_dir`:
  model.txt           LightGBM booster
  candidates.parquet  one row per (customer, candidate article) with all ranking features
  popularity.json     global most-popular articles (fallback for unknown customers)
  meta.json           feature order, cutoff, sizes, settings
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from hm_recsys.baselines import popularity
from hm_recsys.data import actuals_by_customer, history_before
from hm_recsys.features import add_labels, all_features, build_features, training_cutoffs
from hm_recsys.ranker import train_ranker

FORMAT_VERSION = 1


def labelled_week_frame(
    df: pd.DataFrame,
    customers: pd.DataFrame,
    articles: pd.DataFrame,
    cutoff: date,
    n: int = 100,
    als_kwargs: dict | None = None,
) -> pd.DataFrame | None:
    """Candidates and features as of `cutoff` for the customers who bought in the 7 days from
    `cutoff`, labelled with those purchases. None if nobody bought. Customer ids are tagged with
    the week so the same customer is a separate ranking group in each training week."""
    t = pd.to_datetime(df["t_dat"]).dt.date
    week = df[(t >= cutoff) & (t < cutoff + timedelta(days=7))]
    actuals = actuals_by_customer(week)
    if not actuals:
        return None
    frame = build_features(df, customers, articles, cutoff, list(actuals), n, als_kwargs)
    frame = add_labels(frame, actuals)
    frame["customer_id"] = frame["customer_id"] + "@" + str(cutoff)
    return frame


def build_artifacts(
    df: pd.DataFrame,
    customers: pd.DataFrame,
    articles: pd.DataFrame,
    cutoff: date,
    serve_ids: list,
    out_dir: Path,
    train_weeks: int = 4,
    n_candidates: int = 100,
    ranker_params: dict | None = None,
    als_kwargs: dict | None = None,
    seed: int = 0,
    extra_meta: dict | None = None,
) -> dict:
    """Train the ranker on the `train_weeks` weeks before `cutoff` and write the artifacts for
    `serve_ids` as of `cutoff` (features use purchases before `cutoff` only).

    Returns {"meta", "model" (the sklearn-style ranker), "candidates" (the serving frame)} so
    callers can check the served ranking against the offline one."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [
        labelled_week_frame(df, customers, articles, c, n_candidates, als_kwargs)
        for c in training_cutoffs(cutoff, train_weeks)
    ]
    train = pd.concat([f for f in frames if f is not None], ignore_index=True)
    features = all_features()
    model = train_ranker(train, features, {**(ranker_params or {}), "random_state": seed})

    cand = build_features(df, customers, articles, cutoff, serve_ids, n_candidates, als_kwargs)
    model.booster_.save_model(str(out_dir / "model.txt"))
    cand.to_parquet(out_dir / "candidates.parquet", index=False)
    pop = popularity(df, cutoff, n_candidates)
    (out_dir / "popularity.json").write_text(json.dumps([int(a) for a in pop]))

    served = sorted(cand["customer_id"].unique())
    meta = {
        "format_version": FORMAT_VERSION,
        "cutoff": str(cutoff),
        "history_rows": int(len(history_before(df, cutoff))),
        "features": features,
        "n_features": len(features),
        "n_customers": len(served),
        "n_candidate_rows": int(len(cand)),
        "train_weeks": train_weeks,
        "train_rows": int(len(train)),
        "train_positives": int(train["label"].sum()),
        "n_candidates_per_source": n_candidates,
        "ranker_params": ranker_params or {},
        "seed": seed,
        "lightgbm_version": lgb.__version__,
        "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "example_customer_ids": served[:3],
        **(extra_meta or {}),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return {"meta": meta, "model": model, "candidates": cand}
