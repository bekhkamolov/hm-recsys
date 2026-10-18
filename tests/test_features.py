from datetime import date, timedelta

import numpy as np
import pandas as pd
import pandas.testing as pdt
import scipy.sparse as sp

from hm_recsys.features import (
    CATEGORICAL,
    GROUPS,
    _lookup_sparse,
    add_labels,
    all_features,
    build_features,
    candidate_frame,
    training_cutoffs,
)
from hm_recsys.ranker import feature_importance, predict_top_k, train_ranker

CUTOFF = date(2020, 8, 31)
SMALL_ALS = {"factors": 4, "iterations": 3}


def _meta():
    customers = pd.DataFrame(
        {"customer_id": [f"c{i}" for i in range(6)], "age": [20, 30, 40, 50, 60, None]}
    )
    articles = pd.DataFrame(
        {
            "article_id": [100, 101, 102, 103, 104, 999],
            "product_type_no": [1, 1, 2, 2, 3, 3],
            "product_group_name": ["a", "a", "b", "b", "c", "c"],
            "graphical_appearance_no": [1] * 6,
            "colour_group_code": [9, 9, 10, 10, 11, 11],
            "department_no": [5, 5, 6, 6, 7, 7],
            "index_group_no": [1, 1, 1, 1, 2, 2],
        }
    )
    return customers, articles


def test_features_do_not_see_the_prediction_window(transactions):
    customers, articles = _meta()
    future = pd.DataFrame(
        {
            "t_dat": [CUTOFF, CUTOFF + timedelta(days=2), CUTOFF + timedelta(days=4)] * 2,
            "customer_id": ["c0", "c1", "c2", "c3", "c4", "c5"],
            "article_id": [999] * 6,
            "price": [500.0] * 6,
            "sales_channel_id": [1] * 6,
        }
    )
    past = transactions[pd.to_datetime(transactions["t_dat"]).dt.date < CUTOFF]
    both = pd.concat([transactions, future], ignore_index=True)
    ids = [f"c{i}" for i in range(6)]
    a = build_features(past, customers, articles, CUTOFF, ids, n=3, als_kwargs=SMALL_ALS)
    b = build_features(both, customers, articles, CUTOFF, ids, n=3, als_kwargs=SMALL_ALS)
    assert not a.empty
    assert 999 not in set(b["article_id"])
    pdt.assert_frame_equal(a, b)


def test_feature_columns_and_groups_are_consistent(transactions):
    customers, articles = _meta()
    frame = build_features(
        transactions, customers, articles, CUTOFF, ["c0", "c1"], n=3, als_kwargs=SMALL_ALS
    )
    assert list(frame.columns) == ["customer_id", "article_id", *all_features()]
    assert len(all_features()) == len(set(all_features()))
    assert set(CATEGORICAL) <= set(GROUPS["article"])
    assert frame[["customer_id", "article_id"]].duplicated().sum() == 0
    assert (frame["p_n_bought"] >= 0).all()


def test_training_weeks_end_before_validation():
    val_start = date(2020, 9, 9)
    cutoffs = training_cutoffs(val_start, 3)
    assert cutoffs == [date(2020, 9, 2), date(2020, 8, 26), date(2020, 8, 19)]
    assert all(c + timedelta(days=7) <= val_start for c in cutoffs)


def test_candidate_frame_and_labels():
    frame = candidate_frame({"x": {"a": [1, 2]}, "y": {"a": [2, 3], "b": [9]}})
    assert list(frame["article_id"]) == [1, 2, 3, 9]
    row = frame[frame["article_id"] == 2].iloc[0]
    assert row["rank_x"] == 2 and row["rank_y"] == 1
    assert np.isnan(frame[frame["article_id"] == 1]["rank_y"].iloc[0])
    labelled = add_labels(frame, {"a": {2, 5}, "b": {9}})
    assert list(labelled["label"]) == [0, 1, 0, 1]


def test_lookup_sparse():
    m = sp.csr_matrix(np.array([[0, 2.0, 0], [0, 0, 0], [3.0, 0, 1.0]]))
    out = _lookup_sparse(m, np.array([0, 0, 2, 2, 1, -1]), np.array([1, 0, 0, 2, 1, 0]))
    assert out[:5].tolist() == [2.0, 0.0, 3.0, 1.0, 0.0]
    assert np.isnan(out[5])


def test_ranker_learns_a_signal_and_returns_top_k():
    rng = np.random.default_rng(0)
    n_c, per = 40, 8
    frame = pd.DataFrame(
        {
            "customer_id": np.repeat([f"c{i}" for i in range(n_c)], per),
            "article_id": np.tile(np.arange(per), n_c),
            "f_signal": rng.normal(size=n_c * per),
            "f_noise": rng.normal(size=n_c * per),
        }
    )
    frame["label"] = (frame["f_signal"] > 1.0).astype(int)
    feats = ["f_signal", "f_noise"]
    model = train_ranker(frame, feats, {"n_estimators": 30, "min_child_samples": 2})
    top = predict_top_k(model, frame, feats, k=3)
    assert all(len(v) <= 3 for v in top.values())
    imp = feature_importance(model, feats)
    assert next(iter(imp)) == "f_signal"
    positives = frame[frame["label"] == 1].groupby("customer_id")["article_id"].apply(set)
    hit_rate = np.mean([bool(set(top[c]) & items) for c, items in positives.items()])
    assert hit_rate > 0.95
