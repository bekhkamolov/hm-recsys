from datetime import date

import pandas as pd

from hm_recsys.candidates import (
    als_candidates,
    build_interactions,
    buy_again_candidates,
    candidate_recall,
    copurchase_candidates,
    item_similarity,
    popularity_candidates,
    union,
)

CUTOFF = date(2020, 9, 1)


def _tx(rows):
    return pd.DataFrame(rows, columns=["t_dat", "customer_id", "article_id"])


def _history():
    # Customers a, b, c all bought article 1 and 2 together; d bought only 1 and 3 with c.
    rows = []
    for c in ["a", "b", "c"]:
        rows += [(date(2020, 8, 10), c, 1), (date(2020, 8, 11), c, 2)]
    rows += [(date(2020, 8, 12), "c", 3), (date(2020, 8, 20), "d", 1)]
    return _tx(rows)


def test_build_interactions_counts_and_ignores_future():
    df = pd.concat([_history(), _tx([(CUTOFF, "a", 99)])], ignore_index=True)
    inter = build_interactions(df, CUTOFF)
    assert 99 not in inter.articles
    a, one = inter.customer_index["a"], inter.article_index[1]
    assert inter.matrix[a, one] == 1.0


def test_item_similarity_is_cosine_over_customers():
    inter = build_interactions(_history(), CUTOFF)
    sim = item_similarity(inter, neighbors=10)
    i1, i2, i3 = (inter.article_index[a] for a in (1, 2, 3))
    # article 1 bought by 4 customers, 2 by 3, both by 3: 3 / sqrt(4 * 3)
    assert abs(sim[i1, i2] - 3 / (4 * 3) ** 0.5) < 1e-6
    assert sim[i2, i2] == 0
    assert sim[i2, i3] > 0  # c bought both


def test_copurchase_suggests_unbought_articles_and_excludes_owned():
    out = copurchase_candidates(_history(), CUTOFF, ["a", "d", "new"], n=5)
    assert 2 in out["d"] and 1 not in out["d"]  # d owns 1, so gets 2 and 3
    assert set(out["a"]) == {3}  # a owns 1 and 2
    assert "new" not in out  # no history, no co-purchase candidates


def test_sources_do_not_see_the_prediction_window():
    past = _history()
    future = _tx([(CUTOFF, "a", 99), (date(2020, 9, 3), "d", 99), (date(2020, 9, 4), "b", 99)])
    both = pd.concat([past, future], ignore_index=True)
    ids = ["a", "b", "c", "d"]
    assert popularity_candidates(both, CUTOFF, ids) == popularity_candidates(past, CUTOFF, ids)
    assert buy_again_candidates(both, CUTOFF, ids) == buy_again_candidates(past, CUTOFF, ids)
    assert copurchase_candidates(both, CUTOFF, ids) == copurchase_candidates(past, CUTOFF, ids)
    assert als_candidates(both, CUTOFF, ids, n=3, factors=4, iterations=3) == als_candidates(
        past, CUTOFF, ids, n=3, factors=4, iterations=3
    )


def test_als_returns_unbought_articles_only_for_known_customers():
    out = als_candidates(_history(), CUTOFF, ["a", "d", "new"], n=3, factors=4, iterations=3)
    assert "new" not in out
    assert not set(out["a"]) & {1, 2}
    assert not set(out["d"]) & {1}


def test_union_keeps_priority_order_and_removes_duplicates():
    sources = {"x": {"a": [1, 2]}, "y": {"a": [2, 3], "b": [9]}}
    assert union(sources, ["a", "b", "c"]) == {"a": [1, 2, 3], "b": [9], "c": []}
    assert union(sources, ["a"], order=["y", "x"]) == {"a": [2, 3, 1]}


def test_candidate_recall():
    cands = {"a": [1, 2, 3, 4], "b": [9]}
    actuals = {"a": {1, 4}, "b": {5}, "c": {7}}
    full = candidate_recall(cands, actuals)
    assert abs(full["recall"] - (1.0 + 0 + 0) / 3) < 1e-9
    assert abs(full["any_hit"] - 1 / 3) < 1e-9
    assert abs(full["avg_size"] - 5 / 3) < 1e-9
    top2 = candidate_recall(cands, actuals, k=2)
    assert abs(top2["recall"] - 0.5 / 3) < 1e-9
