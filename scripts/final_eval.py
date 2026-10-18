"""Final evaluation of popularity, buy again and the ranker on a target week (Day 10).

    python scripts/final_eval.py --target val    # dry run on the validation week: must reproduce
                                                 # the numbers already in the README
    python scripts/final_eval.py --target test   # the one evaluation of the held-out test week

Everything is fixed before looking at the target week: the ranker has the headline settings (4
training weeks, 300 trees, learning rate 0.05, no tuning, no early stopping), it is trained on the
4 weeks that end right before the target week (so the test run uses the validation week as a
training week but never the test week), and three seeds are all reported. Writes
results/final_eval_<target>.json.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # recommended by implicit

import argparse
import gc
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from hm_recsys.artifacts import labelled_week_frame
from hm_recsys.baselines import fill, own_purchases, popularity, segment_labels
from hm_recsys.data import actuals_by_customer, make_split, split_transactions
from hm_recsys.features import add_labels, all_features, build_features, training_cutoffs
from hm_recsys.metrics import average_precision_at_k, evaluate, evaluate_by_segment
from hm_recsys.ranker import predict_top_k, train_ranker
from hm_recsys.tracking import log_results_json

ROOT = Path(__file__).resolve().parents[1]
TRAIN_WEEKS = 4
SEEDS = 3
K = 12


def timed(label, fn):
    t0 = time.time()
    out = fn()
    print(f"{label}: {time.time() - t0:.0f}s", flush=True)
    return out


def bootstrap_diff(preds_a: dict, preds_b: dict, actuals: dict, n_boot: int = 2000) -> dict:
    """Paired bootstrap over customers of mean AP@12 of model A minus model B."""
    diffs = np.array(
        [
            average_precision_at_k(preds_a.get(c, []), a, K)
            - average_precision_at_k(preds_b.get(c, []), a, K)
            for c, a in actuals.items()
        ]
    )
    rng = np.random.default_rng(0)
    boot = diffs[rng.integers(0, len(diffs), (n_boot, len(diffs)))].mean(axis=1)
    return {
        "mean_diff": float(diffs.mean()),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "share_of_resamples_above_zero": float((boot > 0).mean()),
        "resamples": n_boot,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["val", "test"], required=True)
    args = p.parse_args()

    processed = ROOT / "data" / "processed"
    info = json.loads((processed / "sample.json").read_text())
    df = pd.read_parquet(processed / "transactions.parquet")
    customers = pd.read_parquet(processed / "customers.parquet")
    articles = pd.read_parquet(processed / "articles.parquet")
    split = make_split(date.fromisoformat(info["last_date"]))
    cutoff = split.val_start if args.target == "val" else split.test_start
    window_end = split.val_end if args.target == "val" else split.test_end
    actuals = actuals_by_customer(split_transactions(df, split)[args.target])
    ids = list(actuals)
    features = all_features()
    print(f"target {args.target}: {cutoff} to {window_end}, {len(ids)} customers", flush=True)

    results: dict = {
        "sample": info,
        "target": args.target,
        "target_window": [str(cutoff), str(window_end)],
        "n_customers": len(ids),
        "n_purchases": sum(len(v) for v in actuals.values()),
        "training_week_cutoffs": [str(c) for c in training_cutoffs(cutoff, TRAIN_WEEKS)],
    }

    # baselines: use only purchases before the target week
    top = popularity(df, cutoff, K)
    global_pop = {c: top for c in ids}
    buy_again = fill(own_purchases(df, cutoff, K), global_pop, ids, K)
    models = {"popularity": global_pop, "buy again + popularity": buy_again}

    # the ranker: candidates and features for the target week, trained on the 4 weeks before it
    target_frame = add_labels(
        timed(
            "features for target week", lambda: build_features(df, customers, articles, cutoff, ids)
        ),
        actuals,
    )
    results["candidates"] = {
        "rows": len(target_frame),
        "purchased_rows": int(target_frame["label"].sum()),
        "share_of_purchases_in_candidates": float(target_frame["label"].sum())
        / results["n_purchases"],
    }
    weeks = [
        timed(f"training week {c}", lambda c=c: labelled_week_frame(df, customers, articles, c))
        for c in training_cutoffs(cutoff, TRAIN_WEEKS)
    ]
    train = pd.concat(weeks, ignore_index=True)
    del weeks
    gc.collect()
    results["train_rows"] = len(train)
    results["train_positives"] = int(train["label"].sum())

    ranker_preds = {}
    for seed in range(SEEDS):
        model = timed(
            f"ranker seed {seed}",
            lambda sd=seed: train_ranker(train, features, {"random_state": sd}),
        )
        ranker_preds[seed] = predict_top_k(model, target_frame, features, K)
        models[f"ranker seed {seed}"] = ranker_preds[seed]
        del model
        gc.collect()

    results["overall"] = {name: evaluate(p, actuals, K) for name, p in models.items()}
    seeds = [results["overall"][f"ranker seed {s}"] for s in range(SEEDS)]
    results["ranker_mean_over_seeds"] = {
        m: float(np.mean([r[m] for r in seeds])) for m in ("map@12", "recall@12", "ndcg@12")
    }
    results["bootstrap_map_ranker_minus_buy_again"] = {
        f"seed {s}": bootstrap_diff(ranker_preds[s], buy_again, actuals) for s in range(SEEDS)
    }

    seg = segment_labels(df, customers, cutoff, ids)
    for dim in ("history", "age"):
        labels = {c: s[dim] for c, s in seg.items()}
        shown = {
            "popularity": global_pop,
            "buy again + popularity": buy_again,
            "ranker seed 0": ranker_preds[0],
        }
        results[f"by_{dim}"] = {
            name: evaluate_by_segment(p, actuals, labels, K) for name, p in shown.items()
        }

    (ROOT / "results").mkdir(exist_ok=True)
    out = ROOT / "results" / f"final_eval_{args.target}.json"
    out.write_text(json.dumps(results, indent=2))
    log_results_json(
        out,
        "hm-recsys/final-eval",
        run_name=f"final_eval_{args.target}",
        params={"day": "10", "target": args.target},
    )

    print(f"\n{args.target} week {cutoff} to {window_end}: {len(ids)} customers")
    print("| model | MAP@12 | recall@12 | NDCG@12 |\n|---|---|---|---|")
    for name, m in results["overall"].items():
        print(f"| {name} | {m['map@12']:.4f} | {m['recall@12']:.4f} | {m['ndcg@12']:.4f} |")
    print(
        "ranker mean over seeds:",
        {k: round(v, 4) for k, v in results["ranker_mean_over_seeds"].items()},
    )
    print(
        "bootstrap:",
        {
            k: (round(v["mean_diff"], 4), [round(x, 4) for x in v["ci95"]])
            for k, v in results["bootstrap_map_ranker_minus_buy_again"].items()
        },
    )
    print("candidates:", results["candidates"])


if __name__ == "__main__":
    main()
