import json
import subprocess
import sys
from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from hm_recsys.api import create_app
from hm_recsys.artifacts import build_artifacts
from hm_recsys.baselines import popularity
from hm_recsys.demo import synthetic_meta, synthetic_transactions
from hm_recsys.features import all_features
from hm_recsys.ranker import predict_top_k

CUTOFF = date(2020, 8, 31)
SMALL = {
    "ranker_params": {"n_estimators": 25, "min_child_samples": 5, "num_leaves": 7},
    "als_kwargs": {"factors": 8, "iterations": 3},
    "n_candidates": 15,
    "train_weeks": 3,
}


@pytest.fixture(scope="module")
def world():
    tx = synthetic_transactions(n_customers=150, weeks=10, seed=0)
    customers, articles = synthetic_meta(tx)
    history = tx[pd.to_datetime(tx["t_dat"]).dt.date < CUTOFF]
    serve_ids = sorted(history["customer_id"].unique())[:40]
    return tx, customers, articles, serve_ids


@pytest.fixture(scope="module")
def built(world, tmp_path_factory):
    tx, customers, articles, serve_ids = world
    out = tmp_path_factory.mktemp("artifacts")
    result = build_artifacts(tx, customers, articles, CUTOFF, serve_ids, out, **SMALL)
    return out, result


@pytest.fixture(scope="module")
def client(built):
    with TestClient(create_app(built[0])) as c:
        yield c


def test_artifacts_are_written_and_described(built, world):
    out, result = built
    assert {p.name for p in out.iterdir()} == {
        "model.txt",
        "candidates.parquet",
        "popularity.json",
        "meta.json",
    }
    meta = json.loads((out / "meta.json").read_text())
    assert meta["cutoff"] == "2020-08-31" and meta["features"] == all_features()
    assert meta["n_customers"] == 40 and meta["train_positives"] > 0
    assert meta["example_customer_ids"] == sorted(world[3])[:3]


def test_health_reports_the_loaded_snapshot(client):
    body = client.get("/health").json()
    assert body["ready"] is True and body["status"] == "ok"
    assert body["cutoff"] == "2020-08-31" and body["customers"] == 40
    assert body["n_features"] == len(all_features())


def test_served_ranking_equals_the_offline_ranking(client, built):
    """The service must give exactly what the trained model gives offline for the same rows."""
    _, result = built
    offline = predict_top_k(result["model"], result["candidates"], all_features(), k=12)
    assert len(offline) == 40
    for customer, expected in offline.items():
        body = client.get("/recommend", params={"customer_id": customer}).json()
        assert body["source"] == "ranker"
        assert body["article_ids"] == expected, customer


def test_recommend_response_shape_and_k(client, world):
    customer = world[3][0]
    body = client.get("/recommend", params={"customer_id": customer, "k": 5}).json()
    assert body["customer_id"] == customer
    assert len(body["article_ids"]) == 5 == len(set(body["article_ids"]))
    scores = body["scores"]
    assert scores == sorted(scores, reverse=True)
    default = client.get("/recommend", params={"customer_id": customer}).json()
    assert len(default["article_ids"]) == 12


def test_unknown_customer_gets_the_popularity_fallback(client, world):
    tx = world[0]
    body = client.get("/recommend", params={"customer_id": "no-such-customer", "k": 12}).json()
    assert body["source"] == "popularity_fallback" and body["scores"] is None
    assert body["article_ids"] == popularity(tx, CUTOFF, 12)


def test_stats_count_requests_by_source_and_hold_no_customer_ids(built, world):
    with TestClient(create_app(built[0])) as c:
        empty = c.get("/stats").json()
        assert empty["requests"] == 0 and empty["fallback_share"] is None
        assert empty["scoring_ms_mean"] is None
        for customer in world[3][:3]:
            c.get("/recommend", params={"customer_id": customer})
        c.get("/recommend", params={"customer_id": "nobody"})
        c.get("/recommend")  # invalid request: not counted
        s = c.get("/stats").json()
        assert (s["requests"], s["ranker_requests"], s["fallback_requests"]) == (4, 3, 1)
        assert s["fallback_share"] == 0.25
        assert 0 < s["scoring_ms_mean"] <= s["scoring_ms_p95"]
        assert set(s) == {
            "requests",
            "ranker_requests",
            "fallback_requests",
            "fallback_share",
            "scoring_ms_mean",
            "scoring_ms_p95",
            "window",
        }


def test_root_page_describes_the_loaded_model_and_links_to_endpoints(client, built):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    page = response.text
    example = json.loads((built[0] / "meta.json").read_text())["example_customer_ids"][0]
    assert f"/recommend?customer_id={example}" in page
    assert "/docs" in page and "/health" in page and "/stats" in page
    assert "2020-08-31" in page  # snapshot date from the model's metadata
    assert "/" not in client.get("/openapi.json").json()["paths"]  # not part of the API schema


def test_root_page_escapes_metadata_and_works_without_artifacts(tmp_path):
    (tmp_path / "meta.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "cutoff": "2020-01-01",
                "data": "<script>alert(1)</script>",
                "n_customers": 3,
                "n_features": 33,
                "example_customer_ids": ['a"b<c'],
                "features": [],
            }
        )
    )
    from hm_recsys.api import Store, landing_page

    # a Store cannot be built from this partial folder, so render the page from a stand-in
    stand_in = Store.__new__(Store)
    stand_in.meta = json.loads((tmp_path / "meta.json").read_text())
    page = landing_page(stand_in)
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert 'a"b' not in page.split("recommend?customer_id=")[1].split('"')[0]
    with TestClient(create_app(tmp_path / "missing")) as c:
        empty = c.get("/")
        assert empty.status_code == 200 and "not loaded" in empty.text


def test_invalid_requests_are_rejected(client):
    assert client.get("/recommend").status_code == 422
    assert client.get("/recommend", params={"customer_id": ""}).status_code == 422
    assert client.get("/recommend", params={"customer_id": "c0001", "k": 0}).status_code == 422
    assert client.get("/recommend", params={"customer_id": "c0001", "k": 51}).status_code == 422


def test_service_without_artifacts_reports_not_ready(tmp_path):
    with TestClient(create_app(tmp_path / "missing")) as c:
        assert c.get("/health").json() == {
            "status": "ok",
            "ready": False,
            "cutoff": None,
            "customers": None,
            "n_features": None,
            "built_utc": None,
        }
        assert c.get("/recommend", params={"customer_id": "c0001"}).status_code == 503


def test_artifacts_do_not_depend_on_purchases_after_the_cutoff(world, tmp_path):
    """Adding rows from the prediction week must not change the precomputed candidates."""
    tx, customers, articles, serve_ids = world
    past = tx[pd.to_datetime(tx["t_dat"]).dt.date < CUTOFF]
    a = build_artifacts(past, customers, articles, CUTOFF, serve_ids, tmp_path / "a", **SMALL)
    b = build_artifacts(tx, customers, articles, CUTOFF, serve_ids, tmp_path / "b", **SMALL)
    assert len(tx) > len(past)
    pd.testing.assert_frame_equal(a["candidates"], b["candidates"])
    assert (tmp_path / "a" / "popularity.json").read_text() == (
        tmp_path / "b" / "popularity.json"
    ).read_text()
    assert a["meta"]["train_rows"] == b["meta"]["train_rows"]


def test_runtime_module_does_not_import_training_code():
    """The slim serving image has no duckdb, implicit or the training modules."""
    code = (
        "import sys, hm_recsys.api; "
        "bad = [m for m in ('duckdb', 'implicit', 'hm_recsys.features', 'hm_recsys.candidates', "
        "'hm_recsys.ranker', 'hm_recsys.data', 'hm_recsys.artifacts') if m in sys.modules]; "
        "sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
