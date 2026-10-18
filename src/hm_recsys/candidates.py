"""Candidate generation: several sources that each propose articles for a customer.

The ranker (later) re-orders the union of these lists, so what matters here is recall: does the
article the customer buys next week appear in the list at all? Every function takes the full
transactions frame and a `cutoff` date and only uses rows before the cutoff.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.cpu.als import AlternatingLeastSquares

from hm_recsys.baselines import own_purchases, popularity
from hm_recsys.data import history_before


@dataclass(frozen=True)
class Interactions:
    """Binary customer x article matrix built from purchases before a cutoff."""

    matrix: sp.csr_matrix  # rows = customers, columns = articles, value = number of purchases
    customers: list
    articles: list
    customer_index: dict
    article_index: dict


def build_interactions(df: pd.DataFrame, cutoff: date) -> Interactions:
    hist = history_before(df, cutoff)
    counts = hist.groupby(["customer_id", "article_id"]).size().rename("n").reset_index()
    customers = sorted(counts["customer_id"].unique().tolist())
    articles = sorted(counts["article_id"].unique().tolist())
    c_idx = {c: i for i, c in enumerate(customers)}
    a_idx = {a: i for i, a in enumerate(articles)}
    matrix = sp.csr_matrix(
        (
            counts["n"].to_numpy(dtype=np.float32),
            (
                counts["customer_id"].map(c_idx).to_numpy(),
                counts["article_id"].map(a_idx).to_numpy(),
            ),
        ),
        shape=(len(customers), len(articles)),
    )
    return Interactions(matrix, customers, articles, c_idx, a_idx)


def popularity_candidates(df: pd.DataFrame, cutoff: date, ids: Iterable, n: int = 100) -> dict:
    """The n most purchased articles of the last 7 days, the same list for everyone."""
    top = popularity(df, cutoff, n)
    return {c: top for c in ids}


def buy_again_candidates(df: pd.DataFrame, cutoff: date, ids: Iterable, n: int = 100) -> dict:
    """Each customer's own distinct past purchases, most recent first."""
    own = own_purchases(df, cutoff, n)
    return {c: own[c] for c in ids if c in own}


def _top_n_rows(scores: sp.csr_matrix, n: int) -> list[np.ndarray]:
    """Column indices of the n largest entries per row, best first (zeros are never returned)."""
    out = []
    for r in range(scores.shape[0]):
        lo, hi = scores.indptr[r], scores.indptr[r + 1]
        cols, vals = scores.indices[lo:hi], scores.data[lo:hi]
        order = np.lexsort((cols, -vals))[:n]  # ties by column index for determinism
        out.append(cols[order])
    return out


def item_similarity(inter: Interactions, neighbors: int = 50) -> sp.csr_matrix:
    """Cosine similarity between articles over the customers who bought both ("bought together"),
    keeping only the `neighbors` most similar articles per article to keep the matrix small."""
    binary = inter.matrix.copy()
    binary.data[:] = 1.0
    co = (binary.T @ binary).tocsr()
    co.setdiag(0)
    co.eliminate_zeros()
    pop = np.asarray(binary.sum(axis=0)).ravel()
    inv = 1.0 / np.sqrt(np.maximum(pop, 1.0))
    sim = sp.diags(inv) @ co @ sp.diags(inv)
    sim = sim.tocsr()
    rows, cols, vals = [], [], []
    for r, top in enumerate(_top_n_rows(sim, neighbors)):
        lo, hi = sim.indptr[r], sim.indptr[r + 1]
        lookup = dict(zip(sim.indices[lo:hi], sim.data[lo:hi], strict=True))
        for c in top:
            rows.append(r)
            cols.append(c)
            vals.append(lookup[c])
    return sp.csr_matrix((vals, (rows, cols)), shape=sim.shape, dtype=np.float32)


def copurchase_candidates(
    df: pd.DataFrame,
    cutoff: date,
    ids: Iterable,
    n: int = 100,
    recent_items: int = 10,
    neighbors: int = 50,
    inter: Interactions | None = None,
) -> dict:
    """Articles most similar to the customer's `recent_items` latest purchases. Articles the
    customer already bought are excluded (they come from the buy-again source)."""
    inter = inter or build_interactions(df, cutoff)
    sim = item_similarity(inter, neighbors)
    own = own_purchases(df, cutoff, recent_items)
    ids = [c for c in ids if c in own]
    rows, cols = [], []
    for r, c in enumerate(ids):
        for a in own[c]:
            rows.append(r)
            cols.append(inter.article_index[a])
    seeds = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(ids), len(inter.articles)),
    )
    scores = (seeds @ sim).tocsr()
    bought = inter.matrix[[inter.customer_index[c] for c in ids]].tocsr()
    bought.data[:] = 1.0
    scores = (scores - scores.multiply(bought)).tocsr()
    scores.eliminate_zeros()
    tops = _top_n_rows(scores, n)
    return {c: [inter.articles[j] for j in top] for c, top in zip(ids, tops, strict=True)}


def als_candidates(
    df: pd.DataFrame,
    cutoff: date,
    ids: Iterable,
    n: int = 100,
    factors: int = 64,
    iterations: int = 15,
    regularization: float = 0.01,
    seed: int = 0,
    inter: Interactions | None = None,
    filter_bought: bool = True,
) -> dict:
    """Top-n articles by ALS score among articles the customer has not bought. Customers with no
    history before the cutoff get nothing from this source."""
    inter = inter or build_interactions(df, cutoff)
    model = AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        random_state=seed,
        calculate_training_loss=False,
    )
    confidence = inter.matrix.copy()
    confidence.data = np.log1p(confidence.data) + 1.0  # repeat purchases count a bit more
    model.fit(confidence, show_progress=False)
    ids = [c for c in ids if c in inter.customer_index]
    rows = np.array([inter.customer_index[c] for c in ids])
    found, found_scores = model.recommend(
        rows, confidence[rows], N=n, filter_already_liked_items=filter_bought
    )
    # implicit pads with already-bought items (score -FLT_MAX) when fewer than n articles remain
    return {
        c: [inter.articles[j] for j, s in zip(found[i], found_scores[i], strict=True) if s > -1e30]
        for i, c in enumerate(ids)
    }


def union(sources: Mapping[str, Mapping], ids: Iterable, order: list[str] | None = None) -> dict:
    """Merge the sources per customer, keeping the first occurrence; `order` sets priority."""
    names = order or list(sources)
    out = {}
    for c in ids:
        seen, merged = set(), []
        for name in names:
            for a in sources[name].get(c, []):
                if a not in seen:
                    seen.add(a)
                    merged.append(a)
        out[c] = merged
    return out


def candidate_recall(candidates: Mapping, actuals: Mapping, k: int | None = None) -> dict:
    """Mean recall of the candidate lists (first k items, or the whole list when k is None) over
    all customers in `actuals`, the mean list length, and the share of customers with a hit."""
    n = len(actuals)
    recall = hits = size = 0.0
    for c, items in actuals.items():
        cand = candidates.get(c, [])
        cand = cand if k is None else cand[:k]
        found = len(set(cand) & items)
        recall += found / len(items)
        hits += found > 0
        size += len(cand)
    return {"recall": recall / n, "any_hit": hits / n, "avg_size": size / n}
