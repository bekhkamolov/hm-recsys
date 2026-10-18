"""Train and evaluate the LightGBM ranker on the validation week (Days 4-5).

Training data: candidates generated at the start of each of the `--train-weeks` weeks before
validation, labelled with that week's purchases. Validation: candidates generated at the start of
the validation week. The test week is not used. Writes results/ranker_val.json.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # recommended by implicit

import argparse
import gc
import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from hm_recsys.baselines import fill, own_purchases, popularity, segment_labels
from hm_recsys.data import actuals_by_customer, make_split, split_transactions
from hm_recsys.features import (
    GROUPS,
    add_labels,
    all_features,
    build_features,
    training_cutoffs,
)
from hm_recsys.metrics import average_precision_at_k, evaluate, evaluate_by_segment
from hm_recsys.ranker import feature_importance, predict_top_k, train_ranker
from hm_recsys.tracking import log_results_json

ROOT = Path(__file__).resolve().parents[1]
K = 12


def timed(label, fn):
    t0 = time.time()
    out = fn()
    print(f"{label}: {time.time() - t0:.0f}s", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-weeks", type=int, default=2)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--max-weeks", type=int, default=6, help="most training weeks to build")
    args = p.parse_args()
    n_build = max(args.train_weeks, args.max_weeks)

    processed = ROOT / "data" / "processed"
    info = json.loads((processed / "sample.json").read_text())
    df = pd.read_parquet(processed / "transactions.parquet")
    customers = pd.read_parquet(processed / "customers.parquet")
    articles = pd.read_parquet(processed / "articles.parquet")
    split = make_split(date.fromisoformat(info["last_date"]))
    parts = split_transactions(df, split)
    val_actuals = actuals_by_customer(parts["val"])
    val_ids = list(val_actuals)

    train_frames = []
    for cutoff in training_cutoffs(split.val_start, n_build):  # most recent week first
        week = df[
            (pd.to_datetime(df["t_dat"]).dt.date >= cutoff)
            & (pd.to_datetime(df["t_dat"]).dt.date < cutoff + timedelta(days=7))
        ]
        actuals = actuals_by_customer(week)
        frame = timed(
            f"features for training week {cutoff}",
            lambda c=cutoff, a=actuals: build_features(df, customers, articles, c, list(a)),
        )
        labelled = add_labels(frame, actuals)
        # the same customer is a separate ranking group in each week
        labelled["customer_id"] = labelled["customer_id"] + "@" + str(cutoff)
        train_frames.append(labelled)
    train = pd.concat(train_frames[: args.train_weeks], ignore_index=True)
    val = add_labels(
        timed(
            f"features for validation week {split.val_start}",
            lambda: build_features(df, customers, articles, split.val_start, val_ids),
        ),
        val_actuals,
    )
    print(
        f"train rows {len(train):,} positives {int(train['label'].sum()):,}; "
        f"val rows {len(val):,} positives {int(val['label'].sum()):,}",
        flush=True,
    )

    def run(features, seed=0):
        model = train_ranker(train, features, {"random_state": seed})
        preds = predict_top_k(model, val, features, K)
        return model, preds, evaluate(preds, val_actuals, K)

    feats = all_features()
    results = {
        "sample": info,
        "eval_window": [str(split.val_start), str(split.val_end)],
        "n_val_customers": len(val_ids),
        "train_weeks": args.train_weeks,
        "train_rows": len(train),
        "train_positives": int(train["label"].sum()),
        "val_rows": len(val),
        "val_positives": int(val["label"].sum()),
        "val_positives_total": sum(len(v) for v in val_actuals.values()),
    }

    # reference: the Day 2 best baseline, recomputed here on the same population
    own = own_purchases(df, split.val_start, K)
    top = popularity(df, split.val_start, K)
    base = fill(own, {c: top for c in val_ids}, val_ids, K)
    results["baseline_buy_again_global_pop"] = evaluate(base, val_actuals, K)

    seeds = {}
    for seed in range(args.seeds):
        model, preds, m = timed(f"full model seed {seed}", lambda s=seed: run(feats, s))
        seeds[seed] = m
        if seed == 0:
            full_preds = preds
            results["importance"] = feature_importance(model, feats)
    results["full_model_seeds"] = seeds
    results["full_model"] = seeds[0]

    # paired bootstrap over customers: ranker (seed 0) minus baseline, per-customer AP@12
    rng = np.random.default_rng(0)
    ap_diff = np.array(
        [
            average_precision_at_k(full_preds.get(c, []), a, K)
            - average_precision_at_k(base.get(c, []), a, K)
            for c, a in val_actuals.items()
        ]
    )
    boot = ap_diff[rng.integers(0, len(ap_diff), (2000, len(ap_diff)))].mean(axis=1)
    results["bootstrap_map_diff_vs_baseline"] = {
        "mean_diff": float(ap_diff.mean()),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "share_of_resamples_above_zero": float((boot > 0).mean()),
        "resamples": 2000,
    }

    seg = segment_labels(df, customers, split.val_start, val_ids)
    for dim in ("history", "age"):
        labels = {c: s[dim] for c, s in seg.items()}
        results[f"by_{dim}"] = {
            "ranker": evaluate_by_segment(full_preds, val_actuals, labels, K),
            "baseline": evaluate_by_segment(base, val_actuals, labels, K),
        }

    ablation = {}
    for group in GROUPS:
        keep = [f for f in feats if f not in GROUPS[group]]
        _, _, m = timed(f"without {group}", lambda k=keep: run(k))
        ablation[f"without {group}"] = m
    for group in GROUPS:
        only = GROUPS[group]
        _, _, m = timed(f"only {group}", lambda o=only: run(o))
        ablation[f"only {group}"] = m
    results["ablation"] = ablation

    # Does more training data help? Run last, after freeing the headline training set, because
    # six weeks of candidate rows do not fit in memory next to it on a 16 GB laptop.
    train = None
    gc.collect()
    sensitivity = {}
    for n_weeks in (1, 2, 4, 6):
        if n_weeks > n_build:
            continue
        part = pd.concat(train_frames[:n_weeks], ignore_index=True)
        runs = []
        for seed in range(args.seeds):
            model = train_ranker(part, feats, {"random_state": seed})
            runs.append(evaluate(predict_top_k(model, val, feats, K), val_actuals, K))
            del model
            gc.collect()
        sensitivity[str(n_weeks)] = {
            "train_rows": len(part),
            "train_positives": int(part["label"].sum()),
            "map@12_by_seed": [r["map@12"] for r in runs],
            "recall@12_by_seed": [r["recall@12"] for r in runs],
        }
        print(f"{n_weeks} training weeks: MAP@12 {sensitivity[str(n_weeks)]['map@12_by_seed']}")
        del part
        gc.collect()
    results["training_weeks_sensitivity"] = sensitivity

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "ranker_val.json").write_text(json.dumps(results, indent=2))
    log_results_json(
        ROOT / "results" / "ranker_val.json",
        "hm-recsys/ranker",
        params={"day": "4-5", "train_weeks": str(args.train_weeks), "seeds": str(args.seeds)},
    )

    def row(name, m):
        return f"| {name} | {m['map@12']:.4f} | {m['recall@12']:.4f} | {m['ndcg@12']:.4f} |"

    print("\n| model | MAP@12 | recall@12 | NDCG@12 |\n|---|---|---|---|")
    print(row("buy again + global popularity", results["baseline_buy_again_global_pop"]))
    for s, m in seeds.items():
        print(row(f"ranker, all features, seed {s}", m))
    for name, m in ablation.items():
        print(row(name, m))
    print("\ntop features:", list(results["importance"].items())[:12])
    for dim in ("history", "age"):
        print(f"\nMAP@12 by {dim}")
        for who, per in results[f"by_{dim}"].items():
            print(who, {k: (round(v["map@12"], 4), v["n_customers"]) for k, v in per.items()})


if __name__ == "__main__":
    main()
