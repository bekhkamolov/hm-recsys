from datetime import date

import pandas as pd

from hm_recsys.baselines import (
    age_group,
    age_group_popularity,
    fill,
    own_purchases,
    popularity,
    segment_labels,
)
from hm_recsys.data import prepare_customers
from hm_recsys.metrics import evaluate_by_segment

CUTOFF = date(2020, 9, 1)


def _tx(rows):
    return pd.DataFrame(rows, columns=["t_dat", "customer_id", "article_id"])


def test_popularity_uses_only_the_last_window_before_cutoff():
    df = _tx(
        [
            (date(2020, 8, 1), "a", 1),  # older than 7 days: ignored
            (date(2020, 8, 1), "b", 1),
            (date(2020, 8, 1), "c", 1),
            (date(2020, 8, 28), "a", 2),
            (date(2020, 8, 29), "b", 2),
            (date(2020, 8, 30), "c", 3),
        ]
    )
    assert popularity(df, CUTOFF, k=2) == [2, 3]


def test_baselines_do_not_see_the_prediction_window():
    past = _tx([(date(2020, 8, 28), "a", 1), (date(2020, 8, 29), "b", 1)])
    future = _tx([(CUTOFF, "a", 99), (date(2020, 9, 3), "b", 99), (date(2020, 9, 5), "c", 99)])
    both = pd.concat([past, future], ignore_index=True)
    customers = pd.DataFrame({"customer_id": ["a", "b", "c"], "age": [20.0, 30.0, 40.0]})

    assert popularity(both, CUTOFF, k=3) == popularity(past, CUTOFF, k=3)
    assert own_purchases(both, CUTOFF) == own_purchases(past, CUTOFF)
    assert age_group_popularity(both, customers, CUTOFF) == age_group_popularity(
        past, customers, CUTOFF
    )


def test_own_purchases_most_recent_first_and_distinct():
    df = _tx(
        [
            (date(2020, 8, 1), "a", 1),
            (date(2020, 8, 5), "a", 2),
            (date(2020, 8, 9), "a", 1),  # repeat: counted once, at its latest date
            (date(2020, 8, 3), "b", 7),
        ]
    )
    assert own_purchases(df, CUTOFF) == {"a": [1, 2], "b": [7]}
    assert own_purchases(df, CUTOFF, k=1) == {"a": [1], "b": [7]}


def test_fill_pads_without_duplicates_and_respects_caps():
    primary = {"a": [1, 2, 3], "b": []}
    fallback = {"a": [2, 4, 5, 6], "b": [4, 5], "c": [4, 5]}
    out = fill(primary, fallback, ["a", "b", "c"], k=4)
    assert out == {"a": [1, 2, 3, 4], "b": [4, 5], "c": [4, 5]}
    capped = fill(primary, fallback, ["a"], k=4, max_primary=1)
    assert capped == {"a": [1, 2, 4, 5]}


def test_age_group_buckets_and_unknown():
    ages = pd.Series([18.0, 25.0, 34.0, 54.0, 70.0, None])
    assert list(age_group(ages)) == ["<25", "25-34", "25-34", "45-54", "55+", "unknown"]


def test_age_group_popularity_pads_from_global():
    df = _tx(
        [
            (date(2020, 8, 30), "young", 1),
            (date(2020, 8, 30), "old1", 2),
            (date(2020, 8, 30), "old2", 2),
            (date(2020, 8, 30), "old2", 3),
        ]
    )
    customers = pd.DataFrame({"customer_id": ["young", "old1", "old2"], "age": [20.0, 60.0, 61.0]})
    recs = age_group_popularity(df, customers, CUTOFF, k=3)
    assert recs["<25"] == [1, 2, 3]  # own group has only article 1, rest from global
    assert recs["55+"] == [2, 3, 1]


def test_segment_labels_and_evaluation_by_segment():
    df = _tx([(date(2020, 8, 20), "a", 1)])
    customers = pd.DataFrame({"customer_id": ["a", "b"], "age": [30.0, None]})
    seg = segment_labels(df, customers, CUTOFF, ["a", "b", "z"])
    assert seg["a"] == {"history": "returning", "age": "25-34"}
    assert seg["b"] == {"history": "no history", "age": "unknown"}
    assert seg["z"]["age"] == "unknown"

    preds = {"a": [1], "b": [9]}
    actuals = {"a": {1}, "b": {1}}
    by = evaluate_by_segment(preds, actuals, {c: s["history"] for c, s in seg.items()})
    assert by["returning"]["map@12"] == 1.0
    assert by["no history"]["map@12"] == 0.0
    assert by["returning"]["n_customers"] == 1


def test_prepare_customers_keeps_only_sampled_customers(tmp_path):
    (tmp_path / "customers.csv").write_text("customer_id,age\na,30\nb,\nc,50\n")
    pd.DataFrame({"customer_id": ["a", "b"], "t_dat": [1, 2]}).to_parquet(tmp_path / "tx.parquet")
    n = prepare_customers(
        tmp_path / "customers.csv", tmp_path / "tx.parquet", tmp_path / "customers.parquet"
    )
    out = pd.read_parquet(tmp_path / "customers.parquet").sort_values("customer_id")
    assert n == 2
    assert list(out["customer_id"]) == ["a", "b"]
    assert out["age"].isna().tolist() == [False, True]
