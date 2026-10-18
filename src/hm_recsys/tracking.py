"""MLflow tracking helpers (the optional `tracking` extra).

Runs go to a local SQLite database (`mlflow.db` in the working directory) unless
MLFLOW_TRACKING_URI says otherwise; artifacts go to ./mlruns. If mlflow is not installed the helpers
print a note and do nothing, so the experiment scripts still run without it.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

DEFAULT_URI = "sqlite:///mlflow.db"
_INVALID = re.compile(r"[^A-Za-z0-9_\-. :/]")  # characters MLflow allows in metric/param names
_CHUNK = 500


def sanitize(name: str) -> str:
    """Replace characters MLflow rejects (e.g. '@' in 'map@12') and cap the length."""
    return _INVALID.sub("_", name)[:250]


def flatten_numeric(obj, prefix: str = "") -> dict[str, float]:
    """All finite numbers in a nested dict/list, keyed by their dotted path (booleans skipped)."""
    out: dict[str, float] = {}
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            out.update(flatten_numeric(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, list | tuple):
        for i, value in enumerate(obj):
            out.update(flatten_numeric(value, f"{prefix}.{i}"))
    elif isinstance(obj, int | float) and not isinstance(obj, bool) and math.isfinite(obj):
        out[sanitize(prefix)] = float(obj)
    return out


def flatten_params(obj: Mapping, prefix: str = "") -> dict[str, str]:
    """Scalar values of a nested dict as strings, keyed by their dotted path."""
    out: dict[str, str] = {}
    for key, value in obj.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(flatten_params(value, name))
        elif isinstance(value, str | int | float | bool) or value is None:
            out[sanitize(name)] = str(value)[:500]
    return out


def _chunks(items: dict, size: int = _CHUNK) -> Iterable[dict]:
    keys = list(items)
    for i in range(0, len(keys), size):
        yield {k: items[k] for k in keys[i : i + size]}


def log_run(
    experiment: str,
    run_name: str,
    params: Mapping[str, str] | None = None,
    metrics: Mapping[str, float] | None = None,
    artifacts: Iterable[str | Path] = (),
    tags: Mapping[str, str] | None = None,
) -> str | None:
    """Log one run. Returns the run id, or None when mlflow is not installed."""
    try:
        import mlflow
    except ImportError:
        print("mlflow is not installed, so this run is not tracked (pip install '.[tracking]')")
        return None
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_URI))
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        for chunk in _chunks(dict(params or {}), 100):
            mlflow.log_params(chunk)
        for chunk in _chunks(dict(metrics or {})):
            mlflow.log_metrics(chunk)
        for path in artifacts:
            mlflow.log_artifact(str(path))
        if tags:
            mlflow.set_tags(dict(tags))
        return run.info.run_id


def log_results_json(
    path: str | Path,
    experiment: str,
    run_name: str | None = None,
    params: Mapping[str, str] | None = None,
) -> str | None:
    """Log a results JSON written by one of the scripts: its `sample` block becomes params, every
    other number becomes a metric, and the file itself is attached as an artifact."""
    path = Path(path)
    data = json.loads(path.read_text())
    sample = data.pop("sample", {})
    all_params = {**flatten_params(sample, "sample"), **(params or {})}
    return log_run(
        experiment,
        run_name or path.stem,
        params=all_params,
        metrics=flatten_numeric(data),
        artifacts=[path],
    )
