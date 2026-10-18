"""LightGBM LambdaRank over the candidate rows built in `features.py`."""

from __future__ import annotations

from collections.abc import Sequence

import lightgbm as lgb
import pandas as pd

from hm_recsys.features import CATEGORICAL

DEFAULT_PARAMS = {
    "objective": "lambdarank",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 0,
    "verbose": -1,
}


def _group_sizes(frame: pd.DataFrame) -> list[int]:
    return frame.groupby("customer_id", sort=False).size().tolist()


def train_ranker(
    frame: pd.DataFrame, features: Sequence[str], params: dict | None = None
) -> lgb.LGBMRanker:
    """Fit on a labelled frame sorted by customer. Customers with no purchased candidate carry no
    ranking signal for LambdaRank and are dropped to save time."""
    has_positive = frame.groupby("customer_id")["label"].transform("max") > 0
    train = frame[has_positive].sort_values(["customer_id", "article_id"], kind="stable")
    model = lgb.LGBMRanker(**{**DEFAULT_PARAMS, **(params or {})})
    model.fit(
        train[list(features)],
        train["label"],
        group=_group_sizes(train),
        categorical_feature=[f for f in features if f in CATEGORICAL] or "auto",
    )
    return model


def predict_top_k(
    model: lgb.LGBMRanker, frame: pd.DataFrame, features: Sequence[str], k: int = 12
) -> dict:
    """The k highest-scoring candidates per customer, best first."""
    scored = frame[["customer_id", "article_id"]].assign(score=model.predict(frame[list(features)]))
    scored = scored.sort_values(
        ["customer_id", "score", "article_id"], ascending=[True, False, True]
    )
    return scored.groupby("customer_id")["article_id"].apply(lambda s: list(s.head(k))).to_dict()


def feature_importance(model: lgb.LGBMRanker, features: Sequence[str]) -> dict[str, float]:
    """Share of total split gain per feature, largest first."""
    gain = model.booster_.feature_importance(importance_type="gain")
    total = gain.sum() or 1.0
    ranked = sorted(zip(features, gain / total, strict=True), key=lambda kv: -kv[1])
    return {name: float(v) for name, v in ranked}
