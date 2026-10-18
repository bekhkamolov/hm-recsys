"""Subsample the raw transactions to parquet and record the sample description.

Example: python scripts/prepare_data.py --weeks 16 --customer-fraction 0.1
"""

import argparse
import json
from pathlib import Path

from hm_recsys.data import make_split, prepare_transactions

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weeks", type=int, default=16)
    p.add_argument("--customer-fraction", type=float, default=1.0)
    args = p.parse_args()

    info = prepare_transactions(
        ROOT / "data" / "raw" / "transactions_train.csv",
        ROOT / "data" / "processed" / "transactions.parquet",
        weeks=args.weeks,
        customer_fraction=args.customer_fraction,
    )
    from datetime import date

    split = make_split(date.fromisoformat(info["last_date"]))
    info["split"] = {
        "val_start": str(split.val_start),
        "test_start": str(split.test_start),
        "test_end_exclusive": str(split.test_end),
    }
    (ROOT / "data" / "processed" / "sample.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
