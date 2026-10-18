"""Non-learned baselines: popularity, popularity by age group, "buy again" and combinations.

Every function takes the full transactions frame and a `cutoff` date and only looks at rows
before the cutoff (through `history_before`), so a prediction for the window starting at `cutoff`
cannot see that window.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date

import pandas as pd

from hm_recsys.data import history_before

K = 12
AGE_BINS = [0, 25, 35, 45, 55, 200]
AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55+"]
UNKNOWN_AGE = "unknown"


def age_group(age: pd.Series) -> pd.Series:
    """Age bucket per customer; missing ages become 'unknown'."""
    groups = pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS, right=False).astype(object)
    return groups.where(groups.notna(), UNKNOWN_AGE)


def _top_articles(articles: pd.Series, k: int) -> list:
    """Most frequent articles; ties are broken by article id so results are deterministic."""
    counts = articles.value_counts()
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [article for article, _ in ranked[:k]]


def _recent(df: pd.DataFrame, cutoff: date, days: int) -> pd.DataFrame:
    """Rows of the `days` days before `cutoff` (never the cutoff day or later)."""
    hist = history_before(df, cutoff)
    t = pd.to_datetime(hist["t_dat"])
    return hist[t >= pd.Timestamp(cutoff) - pd.Timedelta(days=days)]


def popularity(df: pd.DataFrame, cutoff: date, k: int = K, days: int = 7) -> list:
    """The k most purchased articles in the `days` days before `cutoff`."""
    return _top_articles(_recent(df, cutoff, days)["article_id"], k)


def age_group_popularity(
    df: pd.DataFrame,
    customers: pd.DataFrame,
    cutoff: date,
    k: int = K,
    days: int = 7,
) -> dict[str, list]:
    """Per age group: the k most purchased articles among that group in the last `days` days.

    A group with fewer than k distinct articles is padded from global popularity. `customers` needs
    columns customer_id and age.
    """
    recent = _recent(df, cutoff, days)
    overall = _top_articles(recent["article_id"], k)
    groups = customers.assign(age_group=age_group(customers["age"]))[["customer_id", "age_group"]]
    merged = recent.merge(groups, on="customer_id", how="left")
    merged["age_group"] = merged["age_group"].fillna(UNKNOWN_AGE)
    out = {}
    for group, part in merged.groupby("age_group"):
        top = _top_articles(part["article_id"], k)
        out[group] = _fill(top, overall, k)
    return out


def own_purchases(df: pd.DataFrame, cutoff: date, k: int = K) -> dict:
    """Each customer's own distinct past purchases, most recent first, at most k."""
    hist = history_before(df, cutoff).copy()
    hist["t_dat"] = pd.to_datetime(hist["t_dat"])
    hist = hist.sort_values(["t_dat", "article_id"], ascending=[False, True])
    hist = hist.drop_duplicates(["customer_id", "article_id"])
    return hist.groupby("customer_id")["article_id"].apply(lambda s: list(s.head(k))).to_dict()


def _fill(primary: list, fallback: list, k: int) -> list:
    """`primary` followed by `fallback` items not already present, cut to k."""
    out = list(primary[:k])
    seen = set(out)
    for item in fallback:
        if len(out) >= k:
            break
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def fill(
    primary: Mapping,
    fallback: Mapping,
    customers: Iterable,
    k: int = K,
    max_primary: int | None = None,
) -> dict:
    """For each customer: own `primary` items (at most `max_primary`), then `fallback` items.

    `fallback` must have an entry for every customer in `customers`.
    """
    cap = k if max_primary is None else max_primary
    return {c: _fill(primary.get(c, [])[:cap], fallback[c], k) for c in customers}


def segment_labels(df: pd.DataFrame, customers: pd.DataFrame, cutoff: date, ids: Iterable) -> dict:
    """Segments for the evaluation population.

    `history` is 'returning' when the customer bought something before `cutoff` inside the loaded
    window, otherwise 'no history'. A customer labelled 'no history' may still be an old H&M
    customer who simply did not buy in the window; the data cannot tell us.
    """
    seen = set(history_before(df, cutoff)["customer_id"])
    ages = dict(zip(customers["customer_id"], age_group(customers["age"]), strict=True))
    return {
        c: {
            "history": "returning" if c in seen else "no history",
            "age": ages.get(c, UNKNOWN_AGE),
        }
        for c in ids
    }
