"""Candidate rows and ranking features for one prediction window.

`build_features(df, ..., cutoff, ids)` returns one row per (customer, candidate article) for the
week starting at `cutoff`. Every feature is computed from purchases before `cutoff` only
(through `history_before`); a test appends future purchases and checks nothing changes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from functools import reduce

import numpy as np
import pandas as pd

from hm_recsys.baselines import UNKNOWN_AGE, age_group
from hm_recsys.candidates import (
    als_candidates,
    build_interactions,
    buy_again_candidates,
    copurchase_candidates,
    copurchase_scores,
    fit_als,
    popularity_candidates,
)
from hm_recsys.data import history_before

SOURCES = ["buy_again", "copurchase", "als", "pop"]

# Feature groups, used for the ablation study.
GROUPS: dict[str, list[str]] = {
    "customer": [
        "c_n_purchases",
        "c_n_last28",
        "c_days_since_last",
        "c_days_since_first",
        "c_n_articles",
        "c_mean_price",
        "c_online_share",
        "c_age",
    ],
    "article": [
        "a_pop7",
        "a_pop28",
        "a_pop_all",
        "a_trend",
        "a_mean_price",
        "a_days_since_first",
        "a_days_since_last",
        "a_product_type",
        "a_product_group",
        "a_colour",
        "a_department",
        "a_index_group",
        "a_appearance",
    ],
    "pair_history": [
        "p_n_bought",
        "p_days_since_bought",
        "p_price_ratio",
        "p_type_share",
        "p_colour_share",
    ],
    "age_popularity": ["p_age_pop7"],
    "cf_scores": ["p_als_score", "p_cop_score"],
    "source_ranks": [f"rank_{s}" for s in SOURCES],
}
CATEGORICAL = [
    "a_product_type",
    "a_product_group",
    "a_colour",
    "a_department",
    "a_index_group",
    "a_appearance",
]


def all_features() -> list[str]:
    return [f for cols in GROUPS.values() for f in cols]


def training_cutoffs(val_start: date, weeks: int) -> list[date]:
    """Start dates of the `weeks` weeks right before validation. Each training week ends on or
    before `val_start`, so its labels never come from the validation week."""
    return [val_start - timedelta(days=7 * k) for k in range(1, weeks + 1)]


def _rank_frame(name: str, cands: Mapping) -> pd.DataFrame:
    rows = [(c, a, r) for c, items in cands.items() for r, a in enumerate(items, start=1)]
    return pd.DataFrame(rows, columns=["customer_id", "article_id", f"rank_{name}"])


def candidate_frame(sources: Mapping[str, Mapping]) -> pd.DataFrame:
    """One row per (customer, article) proposed by any source, with the rank in each source
    (NaN where the source did not propose it)."""
    frames = [_rank_frame(name, cands) for name, cands in sources.items()]
    merged = reduce(lambda a, b: a.merge(b, on=["customer_id", "article_id"], how="outer"), frames)
    return merged.sort_values(["customer_id", "article_id"], ignore_index=True)


def _days(cutoff: date, t: pd.Series) -> pd.Series:
    return (pd.Timestamp(cutoff) - t).dt.days


def _customer_features(hist: pd.DataFrame, customers: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    g = hist.groupby("customer_id")
    last28 = hist[hist["t_dat"] >= pd.Timestamp(cutoff) - pd.Timedelta(days=28)]
    out = pd.DataFrame(
        {
            "c_n_purchases": g.size(),
            "c_n_last28": last28.groupby("customer_id").size(),
            "c_days_since_last": _days(cutoff, g["t_dat"].max()),
            "c_days_since_first": _days(cutoff, g["t_dat"].min()),
            "c_n_articles": g["article_id"].nunique(),
            "c_mean_price": g["price"].mean(),
            "c_online_share": (hist["sales_channel_id"] == 2).groupby(hist["customer_id"]).mean(),
        }
    )
    out["c_n_last28"] = out["c_n_last28"].fillna(0)
    ages = customers.drop_duplicates("customer_id").set_index("customer_id")["age"]
    out["c_age"] = ages.reindex(out.index)
    return out


def _article_features(hist: pd.DataFrame, articles: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    t = pd.Timestamp(cutoff)
    g = hist.groupby("article_id")
    pop7 = hist[hist["t_dat"] >= t - pd.Timedelta(days=7)].groupby("article_id").size()
    pop28 = hist[hist["t_dat"] >= t - pd.Timedelta(days=28)].groupby("article_id").size()
    out = pd.DataFrame(
        {
            "a_pop7": pop7,
            "a_pop28": pop28,
            "a_pop_all": g.size(),
            "a_mean_price": g["price"].mean(),
            "a_days_since_first": _days(cutoff, g["t_dat"].min()),
            "a_days_since_last": _days(cutoff, g["t_dat"].max()),
        }
    )
    out[["a_pop7", "a_pop28"]] = out[["a_pop7", "a_pop28"]].fillna(0)
    out["a_trend"] = out["a_pop7"] / (out["a_pop28"] / 4 + 1)
    attrs = articles.drop_duplicates("article_id").set_index("article_id")
    renamed = {
        "product_type_no": "a_product_type",
        "product_group_name": "a_product_group",
        "colour_group_code": "a_colour",
        "department_no": "a_department",
        "index_group_no": "a_index_group",
        "graphical_appearance_no": "a_appearance",
    }
    # dense 0..n-1 category codes (NaN where unknown); the raw ids are arbitrary labels, some huge
    codes = {
        new: attrs[old].astype("category").cat.codes.astype(float).where(attrs[old].notna())
        for old, new in renamed.items()
    }
    return out.join(pd.DataFrame(codes, index=attrs.index), how="left")


def _lookup_sparse(matrix, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """matrix[rows, cols] for many pairs; 0 where the entry is absent, NaN where row < 0."""
    coo = matrix.tocoo()
    n_cols = matrix.shape[1]
    keys = coo.row.astype(np.int64) * n_cols + coo.col
    order = np.argsort(keys)
    keys, data = keys[order], coo.data[order]
    query = np.where(rows >= 0, rows, 0).astype(np.int64) * n_cols + cols
    pos = np.searchsorted(keys, query)
    pos = np.minimum(pos, len(keys) - 1) if len(keys) else pos
    found = (keys[pos] == query) if len(keys) else np.zeros(len(query), dtype=bool)
    out = np.where(found, data[pos] if len(keys) else 0.0, 0.0).astype(np.float64)
    out[rows < 0] = np.nan
    return out


def build_features(
    df: pd.DataFrame,
    customers: pd.DataFrame,
    articles: pd.DataFrame,
    cutoff: date,
    ids: Iterable,
    n: int = 100,
    als_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Candidates (up to `n` per source) and all features for the customers `ids` in the week
    starting at `cutoff`. Returns a frame sorted by customer, ready for labelling."""
    ids = list(ids)
    als_kwargs = als_kwargs or {}
    inter = build_interactions(df, cutoff)
    fitted = fit_als(inter, **als_kwargs)
    sources = {
        "buy_again": buy_again_candidates(df, cutoff, ids, n),
        "copurchase": copurchase_candidates(df, cutoff, ids, n, inter=inter),
        "als": als_candidates(df, cutoff, ids, n, inter=inter, fitted=fitted),
        "pop": popularity_candidates(df, cutoff, ids, n),
    }
    cand = candidate_frame(sources)

    hist = history_before(df, cutoff).copy()
    hist["t_dat"] = pd.to_datetime(hist["t_dat"])
    cust = _customer_features(hist, customers, cutoff)
    art = _article_features(hist, articles, cutoff)
    feats = cand.join(cust, on="customer_id").join(art, on="article_id")

    # customer x article history
    pair = hist.groupby(["customer_id", "article_id"]).agg(
        p_n_bought=("t_dat", "size"), last=("t_dat", "max")
    )
    pair["p_days_since_bought"] = _days(cutoff, pair["last"])
    feats = feats.join(
        pair[["p_n_bought", "p_days_since_bought"]], on=["customer_id", "article_id"]
    )
    feats["p_n_bought"] = feats["p_n_bought"].fillna(0)
    feats["p_price_ratio"] = feats["a_mean_price"] / feats["c_mean_price"]

    # how much of the customer's history is in the same product type / colour as the candidate
    attrs = hist.join(art[["a_product_type", "a_colour"]], on="article_id")
    for col, name in (("a_product_type", "p_type_share"), ("a_colour", "p_colour_share")):
        share = attrs.groupby(["customer_id", col]).size().rename("n").reset_index()
        share = share.merge(cust[["c_n_purchases"]], left_on="customer_id", right_index=True)
        share[name] = share["n"] / share["c_n_purchases"]
        feats = feats.merge(
            share[["customer_id", col, name]], on=["customer_id", col], how="left"
        ).fillna({name: 0.0})

    # popularity of the article among the customer's age group in the last 7 days
    groups = customers.assign(age_group=age_group(customers["age"]))
    groups = groups.drop_duplicates("customer_id").set_index("customer_id")["age_group"]
    recent = hist[hist["t_dat"] >= pd.Timestamp(cutoff) - pd.Timedelta(days=7)]
    recent = recent.assign(age_group=recent["customer_id"].map(groups).fillna(UNKNOWN_AGE))
    age_pop = recent.groupby(["age_group", "article_id"]).size().rename("p_age_pop7").reset_index()
    feats["age_group"] = feats["customer_id"].map(groups).fillna(UNKNOWN_AGE)
    feats = feats.merge(age_pop, on=["age_group", "article_id"], how="left")
    feats["p_age_pop7"] = feats["p_age_pop7"].fillna(0)
    feats = feats.drop(columns="age_group")

    # collaborative scores for every candidate pair, not only for the source's own top n
    model, _ = fitted
    u = feats["customer_id"].map(inter.customer_index).fillna(-1).astype(int).to_numpy()
    i = feats["article_id"].map(inter.article_index).fillna(-1).astype(int).to_numpy()
    als_score = np.full(len(feats), np.nan)
    ok = (u >= 0) & (i >= 0)
    user_f, item_f = np.asarray(model.user_factors), np.asarray(model.item_factors)
    als_score[ok] = np.einsum("ij,ij->i", user_f[u[ok]], item_f[i[ok]])
    feats["p_als_score"] = als_score

    cop_ids, cop = copurchase_scores(df, cutoff, ids, inter)
    row_of = {c: r for r, c in enumerate(cop_ids)}
    rows = feats["customer_id"].map(row_of).fillna(-1).astype(int).to_numpy()
    feats["p_cop_score"] = _lookup_sparse(cop, rows, np.where(i >= 0, i, 0))

    feats = feats.sort_values(["customer_id", "article_id"], ignore_index=True)
    feats = feats[["customer_id", "article_id", *all_features()]]
    feats[all_features()] = feats[all_features()].astype(
        "float32"
    )  # halves memory; LightGBM bins anyway
    return feats


def add_labels(frame: pd.DataFrame, actuals: Mapping) -> pd.DataFrame:
    """Add `label` = 1 where the customer bought the article in the target week."""
    positives = pd.MultiIndex.from_tuples([(c, a) for c, items in actuals.items() for a in items])
    keys = pd.MultiIndex.from_frame(frame[["customer_id", "article_id"]])
    return frame.assign(label=keys.isin(positives).astype(int))
