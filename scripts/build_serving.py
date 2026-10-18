"""Train the headline ranker on the H&M sample and write the serving artifacts.

The snapshot is as of the validation-week start (2020-09-09): the model is trained on the 4 weeks
before it and the candidate rows use purchases before it. The test week is not touched. A
production system would rebuild this weekly. Needs `prepare_data.py` to have run.

    python scripts/build_serving.py --max-customers 2000
    ARTIFACT_DIR=artifacts uvicorn hm_recsys.api:create_app --factory

The ranker is also scored on the whole validation week, so the artifact's model can be compared with
the number in the README (seed 0 of the headline run). The run is logged to MLflow if installed.
"""

import argparse
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # recommended by implicit

import json
from datetime import date
from pathlib import Path

import pandas as pd

from hm_recsys.artifacts import build_artifacts
from hm_recsys.data import actuals_by_customer, history_before, make_split, split_transactions
from hm_recsys.features import add_labels, all_features, build_features
from hm_recsys.metrics import evaluate
from hm_recsys.ranker import predict_top_k
from hm_recsys.tracking import log_run

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "artifacts")
    p.add_argument("--max-customers", type=int, default=2000)
    p.add_argument("--train-weeks", type=int, default=4)
    args = p.parse_args()

    processed = ROOT / "data" / "processed"
    info = json.loads((processed / "sample.json").read_text())
    df = pd.read_parquet(processed / "transactions.parquet")
    customers = pd.read_parquet(processed / "customers.parquet")
    articles = pd.read_parquet(processed / "articles.parquet")
    split = make_split(date.fromisoformat(info["last_date"]))
    cutoff = split.val_start

    # who to precompute for: a reproducible sample of customers with purchases in the last 8 weeks
    # before the cutoff (known at that time; nothing about the coming week is used)
    recent = history_before(df, cutoff)
    recent = recent[pd.to_datetime(recent["t_dat"]).dt.date >= cutoff - pd.Timedelta(days=56)]
    pool = pd.Series(sorted(recent["customer_id"].unique()))
    serve_ids = pool.sample(min(args.max_customers, len(pool)), random_state=0).tolist()

    result = build_artifacts(
        df,
        customers,
        articles,
        cutoff,
        serve_ids,
        args.out,
        train_weeks=args.train_weeks,
        extra_meta={"sample": info, "data": "H&M 10% customer sample"},
    )
    meta = result["meta"]
    print(f"wrote {args.out}: {meta['n_customers']} customers, {meta['n_candidate_rows']:,} rows")

    # offline check of this exact model on the whole validation week (as in scripts/ranker.py)
    val_actuals = actuals_by_customer(split_transactions(df, split)["val"])
    val = add_labels(
        build_features(df, customers, articles, cutoff, list(val_actuals)), val_actuals
    )
    preds = predict_top_k(result["model"], val, all_features(), 12)
    metrics = evaluate(preds, val_actuals, 12)
    print("validation week:", {k: round(v, 4) for k, v in metrics.items()})

    log_run(
        "hm-recsys/serving-artifact",
        f"ranker-cutoff-{cutoff}",
        params={
            "cutoff": str(cutoff),
            "train_weeks": str(args.train_weeks),
            "served_customers": str(meta["n_customers"]),
            "seed": str(meta["seed"]),
            "n_features": str(meta["n_features"]),
            "train_rows": str(meta["train_rows"]),
            "train_positives": str(meta["train_positives"]),
        },
        metrics={f"val_{k.replace('@', '_')}": v for k, v in metrics.items()},
        artifacts=[args.out / "meta.json", args.out / "model.txt"],
        tags={"day": "8"},
    )


if __name__ == "__main__":
    main()
