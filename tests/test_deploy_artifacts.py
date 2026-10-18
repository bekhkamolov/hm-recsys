"""The demo artifacts committed for deployment must match the current code."""

import json
from pathlib import Path

from hm_recsys.api import Store
from hm_recsys.features import all_features

DEMO = Path(__file__).resolve().parents[1] / "deploy" / "demo-artifacts"


def test_committed_demo_artifacts_load_and_match_the_current_features():
    meta = json.loads((DEMO / "meta.json").read_text())
    # if this fails the feature set changed: rebuild with
    #   python scripts/make_demo_artifacts.py --out deploy/demo-artifacts
    assert meta["features"] == all_features()
    assert meta["data"] == "synthetic demo data, not H&M"
    store = Store(DEMO)
    customer = meta["example_customer_ids"][0]
    rec = store.recommend(customer, 12)
    assert rec.source == "ranker" and len(rec.article_ids) == len(set(rec.article_ids)) == 12
    assert store.recommend("nobody", 12).source == "popularity_fallback"
