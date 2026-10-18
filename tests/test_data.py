from datetime import date

import pandas as pd
import pytest

from hm_recsys.data import (
    actuals_by_customer,
    history_before,
    make_split,
    prepare_transactions,
    split_transactions,
)


def test_make_split_windows():
    s = make_split(date(2020, 9, 22))
    assert s.test_start == date(2020, 9, 16)
    assert s.test_end == date(2020, 9, 23)
    assert s.val_start == date(2020, 9, 9)
    assert s.val_end == s.test_start


def test_split_is_disjoint_ordered_and_complete(transactions):
    split = make_split(transactions["t_dat"].max())
    parts = split_transactions(transactions, split)
    assert sum(len(p) for p in parts.values()) == len(transactions)
    assert parts["train"]["t_dat"].max() < parts["val"]["t_dat"].min()
    assert parts["val"]["t_dat"].max() < parts["test"]["t_dat"].min()
    assert (parts["val"]["t_dat"].nunique() <= 7) and (parts["test"]["t_dat"].nunique() <= 7)


def test_no_leakage_features_only_see_the_past(transactions):
    """Anything computed for the validation week must come from history_before(val_start)."""
    split = make_split(transactions["t_dat"].max())
    hist = history_before(transactions, split.val_start)
    val = split_transactions(transactions, split)["val"]
    assert pd.to_datetime(hist["t_dat"]).dt.date.max() < split.val_start
    assert len(hist) > 0
    # no validation or test row may be present in the history
    assert hist.index.intersection(val.index).empty
    assert pd.to_datetime(hist["t_dat"]).dt.date.max() < pd.to_datetime(val["t_dat"]).dt.date.min()


def test_actuals_by_customer():
    df = pd.DataFrame({"customer_id": ["a", "a", "b"], "article_id": [1, 2, 1]})
    assert actuals_by_customer(df) == {"a": {1, 2}, "b": {1}}


def _write_csv(tmp_path, transactions):
    path = tmp_path / "t.csv"
    transactions.to_csv(path, index=False)
    return path


def test_prepare_transactions_window_and_sampling(tmp_path, transactions):
    csv = _write_csv(tmp_path, transactions)
    info = prepare_transactions(csv, tmp_path / "out.parquet", weeks=2)
    assert info["last_date"] == str(transactions["t_dat"].max())
    out = pd.read_parquet(tmp_path / "out.parquet")
    assert pd.to_datetime(out["t_dat"]).dt.date.min() >= date.fromisoformat(info["first_date"])
    assert info["rows"] == len(out)


def test_customer_sample_is_deterministic_and_a_subset(tmp_path, transactions):
    csv = _write_csv(tmp_path, transactions)
    a = prepare_transactions(csv, tmp_path / "a.parquet", weeks=5, customer_fraction=0.5)
    b = prepare_transactions(csv, tmp_path / "b.parquet", weeks=5, customer_fraction=0.5)
    full = prepare_transactions(csv, tmp_path / "f.parquet", weeks=5)
    assert a["customers"] == b["customers"] <= full["customers"]
    # a sampled customer keeps all of their rows
    sampled = pd.read_parquet(tmp_path / "a.parquet")
    kept = set(sampled["customer_id"])
    assert len(sampled) == (transactions["customer_id"].isin(kept)).sum()


def test_invalid_fraction(tmp_path, transactions):
    with pytest.raises(ValueError):
        prepare_transactions(_write_csv(tmp_path, transactions), tmp_path / "x.parquet", 1, 0)
