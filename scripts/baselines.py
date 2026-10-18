"""Evaluate the Day 2 baselines on the validation week and write results/baselines_val.json.

The test week is not touched here. Run after prepare_data.py.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd

from hm_recsys.baselines import (
    UNKNOWN_AGE,
    age_group,
    age_group_popularity,
    fill,
    own_purchases,
    popularity,
    segment_labels,
)
from hm_recsys.data import actuals_by_customer, make_split, split_transactions
from hm_recsys.metrics import evaluate, evaluate_by_segment

ROOT = Path(__file__).resolve().parents[1]
K = 12


def main() -> None:
    processed = ROOT / "data" / "processed"
    info = json.loads((processed / "sample.json").read_text())
    df = pd.read_parquet(processed / "transactions.parquet")
    customers = pd.read_parquet(processed / "customers.parquet")
    split = make_split(date.fromisoformat(info["last_date"]))

    val = split_transactions(df, split)["val"]
    actuals = actuals_by_customer(val)
    ids = list(actuals)
    cutoff = split.val_start

    global_top = popularity(df, cutoff, K)
    global_recs = {c: global_top for c in ids}

    by_group = age_group_popularity(df, customers, cutoff, K)
    groups = dict(zip(customers["customer_id"], age_group(customers["age"]), strict=True))
    age_recs = {c: by_group.get(groups.get(c, UNKNOWN_AGE), global_top) for c in ids}

    own = own_purchases(df, cutoff, K)
    models = {
        "global popularity (last 7 days)": global_recs,
        "popularity by age group (last 7 days)": age_recs,
        "buy again (own purchases, then global popularity)": fill(own, global_recs, ids, K),
        "buy again + age-group popularity": fill(own, age_recs, ids, K),
        "buy again (max 6) + age-group popularity": fill(own, age_recs, ids, K, max_primary=6),
    }

    seg = segment_labels(df, customers, cutoff, ids)
    results = {"sample": info, "eval_window": [str(split.val_start), str(split.val_end)]}
    results["overall"] = {name: evaluate(p, actuals, K) for name, p in models.items()}
    for dim in ("history", "age"):
        labels = {c: s[dim] for c, s in seg.items()}
        results[f"by_{dim}"] = {
            name: evaluate_by_segment(p, actuals, labels, K) for name, p in models.items()
        }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "baselines_val.json").write_text(json.dumps(results, indent=2))

    print(f"validation {results['eval_window']}, {len(ids)} customers")
    print("| model | MAP@12 | recall@12 | NDCG@12 |\n|---|---|---|---|")
    for name, m in results["overall"].items():
        print(f"| {name} | {m['map@12']:.4f} | {m['recall@12']:.4f} | {m['ndcg@12']:.4f} |")
    for dim in ("history", "age"):
        print(f"\nMAP@12 by {dim}")
        for name, per in results[f"by_{dim}"].items():
            print(name, {k: (round(v["map@12"], 4), v["n_customers"]) for k, v in per.items()})


if __name__ == "__main__":
    main()
