"""Log the results/*.json files from earlier runs to MLflow (the `tracking` extra).

Each file becomes one run in its own experiment: the sample description is stored as parameters,
every number as a metric and the JSON as an artifact. Use it for results that already exist; the
experiment scripts also log their own runs when they finish.

    python scripts/track_results.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from pathlib import Path

from hm_recsys.tracking import log_results_json

ROOT = Path(__file__).resolve().parents[1]

RUNS = {
    "baselines_val.json": ("hm-recsys/baselines", {"day": "2"}),
    "candidates_val.json": ("hm-recsys/candidates", {"day": "3"}),
    "ranker_val.json": ("hm-recsys/ranker", {"day": "4-5", "train_weeks": "4"}),
    "ab_cookie_cats.json": ("hm-recsys/ab-cookie-cats", {"day": "6"}),
    "experiment_sim.json": ("hm-recsys/experiment-sim", {"day": "7"}),
}


def main() -> None:
    for name, (experiment, params) in RUNS.items():
        path = ROOT / "results" / name
        if not path.exists():
            print(f"{name}: not found, skipped")
            continue
        run_id = log_results_json(path, experiment, params=params)
        print(f"{name}: logged to {experiment} (run {run_id})")


if __name__ == "__main__":
    main()
