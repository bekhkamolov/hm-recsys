import math

import pytest

from hm_recsys.metrics import average_precision_at_k, evaluate, ndcg_at_k, recall_at_k


def test_ap_perfect_and_empty():
    assert average_precision_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0
    assert average_precision_at_k([4, 5, 6], {1}, k=3) == 0.0
    assert average_precision_at_k([1], set(), k=3) == 0.0


def test_ap_hand_computed():
    # hits at rank 1 and 3: (1/1 + 2/3) / min(2, 3)
    assert average_precision_at_k([1, 9, 2], {1, 2}, k=3) == pytest.approx((1 + 2 / 3) / 2)


def test_ap_denominator_capped_at_k():
    # 5 relevant items but k=2: perfect top-2 must score 1.0, not 0.4
    assert average_precision_at_k([1, 2], {1, 2, 3, 4, 5}, k=2) == 1.0


def test_ap_ignores_duplicates_and_beyond_k():
    assert average_precision_at_k([1, 1, 1], {1}, k=3) == 1.0
    assert average_precision_at_k([9, 9, 1], {1}, k=2) == 0.0


def test_recall():
    assert recall_at_k([1, 2, 9], {1, 2, 3, 4}, k=3) == 0.5
    assert recall_at_k([1, 2, 9], {1, 2, 3, 4}, k=1) == 0.25
    assert recall_at_k([1], set(), k=3) == 0.0


def test_ndcg():
    assert ndcg_at_k([1, 2], {1, 2}, k=2) == pytest.approx(1.0)
    # single relevant item at rank 2: (1/log2(3)) / (1/log2(2))
    assert ndcg_at_k([9, 1], {1}, k=2) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k([9, 8], {1}, k=2) == 0.0


def test_evaluate_missing_prediction_scores_zero():
    actuals = {"a": {1}, "b": {2}}
    res = evaluate({"a": [1]}, actuals, k=12, recall_ks=[100])
    assert res["map@12"] == 0.5
    assert res["recall@100"] == 0.5
    assert res["n_customers"] == 2


def test_evaluate_empty_raises():
    with pytest.raises(ValueError):
        evaluate({}, {})
