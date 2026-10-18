"""Measure candidate-generation recall on the validation week (Day 3).

Writes results/candidates_val.json. The test week is not used. Run after prepare_data.py.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # recommended by implicit

import json
import time
from datetime import date
from pathlib import Path

import pandas as pd

from hm_recsys.candidates import (
    als_candidates,
    build_interactions,
    buy_again_candidates,
    candidate_recall,
    copurchase_candidates,
    popularity_candidates,
    union,
)
from hm_recsys.data import actuals_by_customer, history_before, make_split, split_transactions

ROOT = Path(__file__).resolve().parents[1]
N = 100  # candidates per source and customer
ORDER = ["buy again", "co-purchase", "ALS", "popularity"]


def main() -> None:
    processed = ROOT / "data" / "processed"
    info = json.loads((processed / "sample.json").read_text())
    df = pd.read_parquet(processed / "transactions.parquet")
    split = make_split(date.fromisoformat(info["last_date"]))
    val = split_transactions(df, split)["val"]
    actuals = actuals_by_customer(val)
    ids = list(actuals)
    cutoff = split.val_start

    seen_before = set(history_before(df, cutoff)["article_id"])
    n_items = sum(len(v) for v in actuals.values())
    reachable = sum(len(v & seen_before) for v in actuals.values()) / n_items

    inter = build_interactions(df, cutoff)
    sources, seconds = {}, {}
    builders = {
        "buy again": lambda: buy_again_candidates(df, cutoff, ids, N),
        "co-purchase": lambda: copurchase_candidates(df, cutoff, ids, N, inter=inter),
        "ALS": lambda: als_candidates(df, cutoff, ids, N, inter=inter),
        "popularity": lambda: popularity_candidates(df, cutoff, ids, N),
    }
    for name, build in builders.items():
        t0 = time.time()
        sources[name] = build()
        seconds[name] = round(time.time() - t0, 1)
        print(f"{name}: built in {seconds[name]}s", flush=True)

    history = {c: "returning" if c in set(inter.customer_index) else "no history" for c in ids}
    results = {
        "sample": info,
        "eval_window": [str(split.val_start), str(split.val_end)],
        "n_customers": len(ids),
        "candidates_per_source": N,
        "reachable_share_of_val_purchases": reachable,
        "build_seconds": seconds,
        "per_source": {},
        "union": {},
        "leave_one_out": {},
        "by_history": {},
    }
    for name, cands in sources.items():
        r = candidate_recall(cands, actuals, k=100)
        r["coverage"] = sum(1 for c in ids if cands.get(c)) / len(ids)
        results["per_source"][name] = r

    full = union(sources, ids, ORDER)
    results["union"]["all (no cap)"] = candidate_recall(full, actuals)
    results["union"]["first 200"] = candidate_recall(full, actuals, k=200)
    for name in ORDER:
        rest = [s for s in ORDER if s != name]
        results["leave_one_out"][f"without {name}"] = candidate_recall(
            union(sources, ids, rest), actuals
        )

    for label in ("returning", "no history"):
        part = {c: a for c, a in actuals.items() if history[c] == label}
        results["by_history"][label] = {
            "n_customers": len(part),
            **{
                f"{n} recall@100": candidate_recall(s, part, k=100)["recall"]
                for n, s in sources.items()
            },
            "union recall": candidate_recall(full, part)["recall"],
            "union avg size": candidate_recall(full, part)["avg_size"],
        }

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "candidates_val.json").write_text(json.dumps(results, indent=2))

    print(f"\nvalidation {results['eval_window']}, {len(ids)} customers")
    print(f"share of validation purchases whose article was sold before: {reachable:.3f}")
    print("\n| source | recall@100 | customers with a hit | coverage | avg size |")
    for name, r in results["per_source"].items():
        print(
            f"| {name} | {r['recall']:.4f} | {r['any_hit']:.4f} | {r['coverage']:.4f} "
            f"| {r['avg_size']:.1f} |"
        )
    print("\n| union | recall | customers with a hit | avg size |")
    for name, r in {**results["union"], **results["leave_one_out"]}.items():
        print(f"| {name} | {r['recall']:.4f} | {r['any_hit']:.4f} | {r['avg_size']:.1f} |")
    print("\nby history:", json.dumps(results["by_history"], indent=1))


if __name__ == "__main__":
    main()
