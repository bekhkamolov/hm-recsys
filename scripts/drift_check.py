"""Drift check and model-staleness experiment for the ranker (Day 9).

For the validation-week snapshot (2020-09-09), compares the data the headline model was trained on
with the data it would serve (PSI per feature and for the model score), measures how large PSI is
from sampling noise alone, and measures how much MAP@12 is lost when the model was trained 1, 2 or
3 weeks earlier. The test week is not used. Writes results/drift_val.json.

    python scripts/drift_check.py            # runs every stage below, each in its own process

The work is split into stages that run as separate processes and hand results to the next through
files in data/processed/drift_cache/ (ignored by git). Holding seven weeks of candidate rows and a
training set in one process does not fit in the memory this laptop can commit.
    cache  : weekly candidate/feature frames and the validation frame
    psi    : noise floor, feature PSI (training data vs validation), PSI of each week vs validation
    stale  : train on the 4 weeks ending s weeks before validation (3 seeds), score validation
    assemble: merge everything into results/drift_val.json and log it to MLflow
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # recommended by implicit

import argparse
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from hm_recsys.artifacts import labelled_week_frame
from hm_recsys.data import actuals_by_customer, make_split, split_transactions
from hm_recsys.features import (
    CATEGORICAL,
    GROUPS,
    add_labels,
    all_features,
    build_features,
    training_cutoffs,
)
from hm_recsys.metrics import evaluate
from hm_recsys.monitoring import drift_report, noise_floor, psi, status
from hm_recsys.ranker import predict_top_k, train_ranker
from hm_recsys.tracking import log_results_json

# Arrow's default pool keeps freed memory committed; the system allocator gives it back.
pa.set_memory_pool(pa.system_memory_pool())

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CACHE = PROCESSED / "drift_cache"
TRAIN_WEEKS = 4
MAX_STALE = 3
SEEDS = 3
K = 12
FEATURES = all_features()
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]


def timed(label, fn):
    t0 = time.time()
    out = fn()
    print(f"{label}: {time.time() - t0:.0f}s", flush=True)
    return out


def summarise(report: pd.DataFrame) -> dict:
    return {
        "mean_psi": float(report["psi"].mean()),
        "max_psi": float(report["psi"].max()),
        "features_moderate_or_major": int((report["psi"] >= 0.1).sum()),
        "features_major": int((report["psi"] >= 0.25).sum()),
    }


def setup():
    info = json.loads((PROCESSED / "sample.json").read_text())
    split = make_split(date.fromisoformat(info["last_date"]))
    cutoffs = training_cutoffs(split.val_start, TRAIN_WEEKS + MAX_STALE)  # most recent first
    return info, split, cutoffs


def load_weeks(cutoffs, indices) -> pd.DataFrame:
    """Cached weeks stacked into one frame. Read and joined as Arrow tables so only one pandas copy
    exists; ranking-group ids are offset per week so they stay unique across weeks."""
    tables = []
    for i in indices:
        table = pq.read_table(CACHE / f"week_{cutoffs[i]}.parquet")
        ids = pc.add(table["customer_id"], pa.scalar(i * 10_000_000, pa.int32()))
        col = table.schema.get_field_index("customer_id")
        tables.append(table.set_column(col, "customer_id", ids))
    return pa.concat_tables(tables).to_pandas(split_blocks=True, self_destruct=True)


def val_actuals_for(split) -> dict:
    df = pd.read_parquet(PROCESSED / "transactions.parquet")
    return actuals_by_customer(split_transactions(df, split)["val"])


def stage_cache() -> None:
    info, split, cutoffs = setup()
    df = pd.read_parquet(PROCESSED / "transactions.parquet")
    customers = pd.read_parquet(PROCESSED / "customers.parquet")
    articles = pd.read_parquet(PROCESSED / "articles.parquet")
    for c in cutoffs:
        path = CACHE / f"week_{c}.parquet"
        if path.exists():
            continue
        frame = timed(
            f"features for week {c}", lambda c=c: labelled_week_frame(df, customers, articles, c)
        )
        # long customer hashes -> integer ranking-group ids, unique within a week
        frame["customer_id"] = pd.factorize(frame["customer_id"])[0].astype("int32")
        frame.to_parquet(path, index=False)
    path = CACHE / "val.parquet"
    if not path.exists():
        actuals = actuals_by_customer(split_transactions(df, split)["val"])
        val = timed(
            f"features for validation week {split.val_start}",
            lambda: build_features(df, customers, articles, split.val_start, list(actuals)),
        )
        add_labels(val, actuals).to_parquet(path, index=False)


def stage_psi() -> None:
    info, split, cutoffs = setup()
    val = pd.read_parquet(CACHE / "val.parquet")
    out: dict = {"val_rows": len(val)}

    # how large is PSI from sampling noise alone? split the most recent training week by customer
    # (within one week the integer id is one customer, so each customer lands in one half)
    one = load_weeks(cutoffs, [0])
    floor = noise_floor(one, NUMERIC, CATEGORICAL, groups=one["customer_id"])
    out["noise_floor_customer_halves_one_week"] = {"rows": len(one), "summary": summarise(floor)}
    del one

    # the fresh model's training data vs the validation-week snapshot
    group_of = {f: g for g, cols in GROUPS.items() for f in cols}
    fresh = load_weeks(cutoffs, range(TRAIN_WEEKS))
    t0 = time.time()
    report = drift_report(fresh, val, NUMERIC, CATEGORICAL)
    print(f"feature PSI: {time.time() - t0:.0f}s", flush=True)
    report["group"] = report["feature"].map(group_of)
    out["feature_psi_train_vs_val"] = {
        "training_rows": len(fresh),
        "summary": summarise(report),
        "features": {
            r.feature: {"psi": r.psi, "status": r.status, "group": r.group}
            for r in report.itertuples()
        },
    }
    del fresh

    # how fast does it drift? each earlier week against the validation week
    out["weekly_psi_vs_val"] = {
        str(c): summarise(drift_report(load_weeks(cutoffs, [i]), val, NUMERIC, CATEGORICAL))
        for i, c in enumerate(cutoffs)
    }
    (CACHE / "psi.json").write_text(json.dumps(out, indent=2))


def stage_stale(s: int) -> None:
    info, split, cutoffs = setup()
    val = pd.read_parquet(CACHE / "val.parquet")
    actuals = val_actuals_for(split)
    train = load_weeks(cutoffs, range(s, s + TRAIN_WEEKS))
    runs, first_model = [], None
    for seed in range(SEEDS):
        model = timed(
            f"model trained {s} weeks earlier, seed {seed}",
            lambda sd=seed: train_ranker(train, FEATURES, {"random_state": sd}),
        )
        runs.append(evaluate(predict_top_k(model, val, FEATURES, K), actuals, K))
        if seed == 0:
            first_model = model
    out = {
        "weeks_older": s,
        "training_cutoffs": [str(c) for c in cutoffs[s : s + TRAIN_WEEKS]],
        "map@12_by_seed": [r["map@12"] for r in runs],
        "recall@12_by_seed": [r["recall@12"] for r in runs],
        "ndcg@12_by_seed": [r["ndcg@12"] for r in runs],
        "feature_psi_vs_val": summarise(drift_report(train, val, NUMERIC, CATEGORICAL)),
    }
    if s == 0:
        # score drift of the fresh model: the most recent training week vs the validation week
        # (training rows are in-sample for this model, so the two should look alike at best)
        ref = train[train["customer_id"] < 10_000_000]
        ref_scores = first_model.predict(ref[FEATURES])
        val_scores = first_model.predict(val[FEATURES])
        score_psi = psi(ref_scores, val_scores)
        out["score_psi_last_training_week_vs_val"] = {
            "psi": score_psi,
            "status": status(score_psi),
            "reference_mean": float(ref_scores.mean()),
            "current_mean": float(val_scores.mean()),
            "reference_std": float(ref_scores.std()),
            "current_std": float(val_scores.std()),
        }
    (CACHE / f"stale_{s}.json").write_text(json.dumps(out, indent=2))


def stage_assemble() -> None:
    info, split, cutoffs = setup()
    results: dict = {
        "sample": info,
        "snapshot_cutoff": str(split.val_start),
        "training_week_cutoffs_most_recent_first": [str(c) for c in cutoffs],
        **json.loads((CACHE / "psi.json").read_text()),
    }
    stale = {
        str(s): json.loads((CACHE / f"stale_{s}.json").read_text()) for s in range(MAX_STALE + 1)
    }
    results["score_psi_last_training_week_vs_val"] = stale["0"].pop(
        "score_psi_last_training_week_vs_val"
    )
    results["staleness"] = stale
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "drift_val.json").write_text(json.dumps(results, indent=2))
    log_results_json(ROOT / "results" / "drift_val.json", "hm-recsys/drift", params={"day": "9"})

    print("\nfeature PSI train vs val:", results["feature_psi_train_vs_val"]["summary"])
    print("noise floor:", results["noise_floor_customer_halves_one_week"]["summary"])
    print("score PSI:", results["score_psi_last_training_week_vs_val"])
    print("\nweeks older | MAP@12 by seed | mean PSI | features >= 0.1")
    for s, v in stale.items():
        m = np.mean(v["map@12_by_seed"])
        print(
            f"{s} | {[round(x, 4) for x in v['map@12_by_seed']]} mean {m:.4f} | "
            f"{v['feature_psi_vs_val']['mean_psi']:.3f} | "
            f"{v['feature_psi_vs_val']['features_moderate_or_major']}"
        )
    print(
        "\nweekly PSI vs val:",
        {k: round(v["mean_psi"], 3) for k, v in results["weekly_psi_vs_val"].items()},
    )


def run_stage(*args: str) -> None:
    """Run one stage in a fresh process so its memory is released before the next one starts."""
    cmd = [sys.executable, str(Path(__file__)), *args]
    print(f"--- {' '.join(args)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["all", "cache", "psi", "stale", "assemble"], default="all")
    p.add_argument("--s", type=int, default=0, help="weeks older, for --stage stale")
    args = p.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.stage == "all":
        run_stage("--stage", "cache")
        run_stage("--stage", "psi")
        for s in range(MAX_STALE + 1):
            run_stage("--stage", "stale", "--s", str(s))
        run_stage("--stage", "assemble")
    elif args.stage == "cache":
        stage_cache()
    elif args.stage == "psi":
        stage_psi()
    elif args.stage == "stale":
        stage_stale(args.s)
    else:
        stage_assemble()


if __name__ == "__main__":
    main()
