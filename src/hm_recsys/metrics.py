"""Ranking metrics. All functions take per-customer ranked predictions and sets of purchased items.

Customers with no purchases in the target week are not part of the evaluation population, so
`actual` is expected to contain only customers with at least one purchase.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def average_precision_at_k(predicted: Sequence, actual: set, k: int = 12) -> float:
    """AP@k as defined by the H&M competition: sum of precision@i at each hit, divided by
    min(number of relevant items, k). Duplicate predictions only count once."""
    if not actual:
        return 0.0
    hits = 0
    score = 0.0
    seen = set()
    for i, item in enumerate(predicted[:k], start=1):
        if item in actual and item not in seen:
            hits += 1
            score += hits / i
        seen.add(item)
    return score / min(len(actual), k)


def recall_at_k(predicted: Sequence, actual: set, k: int = 12) -> float:
    """Share of the customer's purchased items that appear in the top k."""
    if not actual:
        return 0.0
    return len(set(predicted[:k]) & actual) / len(actual)


def ndcg_at_k(predicted: Sequence, actual: set, k: int = 12) -> float:
    """NDCG@k with binary relevance and log2 discount."""
    if not actual:
        return 0.0
    seen = set()
    dcg = 0.0
    for i, item in enumerate(predicted[:k], start=1):
        if item in actual and item not in seen:
            dcg += 1.0 / math.log2(i + 1)
        seen.add(item)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(actual), k) + 1))
    return dcg / ideal


def evaluate(
    predictions: Mapping, actuals: Mapping, k: int = 12, recall_ks: Sequence[int] = ()
) -> dict[str, float]:
    """Mean metrics over all customers in `actuals`. A customer without a prediction scores 0."""
    if not actuals:
        raise ValueError("actuals is empty: nothing to evaluate")
    n = len(actuals)
    out = {
        f"map@{k}": 0.0,
        f"recall@{k}": 0.0,
        f"ndcg@{k}": 0.0,
    }
    for extra in recall_ks:
        out[f"recall@{extra}"] = 0.0
    for customer, items in actuals.items():
        pred = list(predictions.get(customer, []))
        out[f"map@{k}"] += average_precision_at_k(pred, items, k)
        out[f"recall@{k}"] += recall_at_k(pred, items, k)
        out[f"ndcg@{k}"] += ndcg_at_k(pred, items, k)
        for extra in recall_ks:
            out[f"recall@{extra}"] += recall_at_k(pred, items, extra)
    result = {name: total / n for name, total in out.items()}
    result["n_customers"] = n
    return result


def evaluate_by_segment(
    predictions: Mapping, actuals: Mapping, segments: Mapping, k: int = 12
) -> dict[str, dict[str, float]]:
    """Metrics per segment. `segments` maps customer -> label; customers without a label are
    grouped as 'unknown'. Returns {label: metrics} including the customer count."""
    groups: dict[str, dict] = {}
    for customer, items in actuals.items():
        groups.setdefault(segments.get(customer, "unknown"), {})[customer] = items
    return {label: evaluate(predictions, part, k) for label, part in sorted(groups.items())}
