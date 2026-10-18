import json
import math

import pytest

from hm_recsys.tracking import (
    flatten_numeric,
    flatten_params,
    log_results_json,
    log_run,
    sanitize,
)

mlflow = pytest.importorskip("mlflow")


def test_sanitize_replaces_characters_mlflow_rejects():
    assert sanitize("map@12") == "map_12"
    assert sanitize("global popularity (last 7 days)") == "global popularity _last 7 days_"
    assert len(sanitize("x" * 400)) == 250


def test_flatten_numeric_keeps_finite_numbers_only():
    nested = {
        "a": {"map@12": 0.5, "n": 3},
        "seeds": [0.1, 0.2],
        "flag": True,
        "name": "text",
        "bad": math.nan,
        "inf": math.inf,
    }
    assert flatten_numeric(nested) == {"a.map_12": 0.5, "a.n": 3.0, "seeds.0": 0.1, "seeds.1": 0.2}


def test_flatten_params_keeps_scalars_with_dotted_names():
    out = flatten_params({"weeks": 16, "split": {"val_start": "2020-09-09", "x": [1]}}, "sample")
    assert out == {"sample.weeks": "16", "sample.split.val_start": "2020-09-09"}


def test_log_run_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # sqlite file and ./mlruns land in the temp folder
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 't.db').as_posix()}")
    artifact = tmp_path / "note.txt"
    artifact.write_text("hello")
    run_id = log_run(
        "test-exp",
        "run-1",
        params={"k": "12"},
        metrics={"map_12": 0.0279},
        artifacts=[artifact],
        tags={"day": "8"},
    )
    run = mlflow.tracking.MlflowClient().get_run(run_id)
    assert run.data.metrics["map_12"] == pytest.approx(0.0279)
    assert run.data.params["k"] == "12" and run.data.tags["day"] == "8"
    names = [f.path for f in mlflow.tracking.MlflowClient().list_artifacts(run_id)]
    assert names == ["note.txt"]


def test_log_results_json_splits_sample_into_params(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 't.db').as_posix()}")
    results = tmp_path / "res.json"
    results.write_text(
        json.dumps({"sample": {"weeks": 16, "rows": 5}, "overall": {"model A": {"map@12": 0.02}}})
    )
    run_id = log_results_json(results, "test-exp", run_name="baselines")
    run = mlflow.tracking.MlflowClient().get_run(run_id)
    assert run.data.params["sample.weeks"] == "16"
    assert run.data.metrics["overall.model A.map_12"] == pytest.approx(0.02)
    assert run.info.run_name == "baselines"
