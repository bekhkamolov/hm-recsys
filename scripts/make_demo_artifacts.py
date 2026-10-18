"""Build serving artifacts from small synthetic data (no H&M download needed).

Used by CI to smoke-test the Docker image and handy for trying the API locally:

    python scripts/make_demo_artifacts.py --out artifacts-demo
    ARTIFACT_DIR=artifacts-demo uvicorn hm_recsys.api:create_app --factory
"""

import argparse
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # recommended by implicit

from datetime import date
from pathlib import Path

import pandas as pd

from hm_recsys.artifacts import build_artifacts
from hm_recsys.demo import synthetic_meta, synthetic_transactions

CUTOFF = date(2020, 8, 31)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("artifacts-demo"))
    p.add_argument("--customers", type=int, default=300)
    args = p.parse_args()

    tx = synthetic_transactions(n_customers=args.customers, weeks=10, seed=0)
    customers, articles = synthetic_meta(tx)
    history = tx[pd.to_datetime(tx["t_dat"]).dt.date < CUTOFF]
    serve_ids = sorted(history["customer_id"].unique())
    result = build_artifacts(
        tx,
        customers,
        articles,
        CUTOFF,
        serve_ids,
        args.out,
        train_weeks=3,
        n_candidates=20,
        ranker_params={"n_estimators": 40, "min_child_samples": 5, "num_leaves": 7},
        als_kwargs={"factors": 8, "iterations": 3},
        extra_meta={"data": "synthetic demo data, not H&M"},
    )
    meta = result["meta"]
    print(f"wrote {args.out}: {meta['n_customers']} customers, {meta['n_candidate_rows']} rows")
    print("example customer ids:", meta["example_customer_ids"])


if __name__ == "__main__":
    main()
