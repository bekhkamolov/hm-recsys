"""FastAPI service: rank precomputed candidates with the LightGBM ranker, fall back to popularity.

Runtime only: this module needs lightgbm, numpy, pandas and fastapi, and imports none of the
training code (a test checks that). Artifacts are built by `hm_recsys.artifacts`.
Run with: uvicorn hm_recsys.api:create_app --factory   (artifact folder from $ARTIFACT_DIR)

  GET /recommend?customer_id=...&k=12   top-k article ids for a customer
  GET /health                           readiness and what snapshot is loaded
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel

SUPPORTED_FORMAT = 1
MAX_K = 50


class Recommendation(BaseModel):
    customer_id: str
    source: Literal["ranker", "popularity_fallback"]
    article_ids: list[int]
    # ranker scores (null where a slot was filled from popularity); null for the fallback source
    scores: list[float | None] | None = None


class Health(BaseModel):
    status: str
    ready: bool
    cutoff: str | None = None
    customers: int | None = None
    n_features: int | None = None
    built_utc: str | None = None


class Store:
    """Loaded artifacts. Candidate rows are kept as one float32 matrix with per-customer slices."""

    def __init__(self, artifact_dir: Path):
        meta = json.loads((artifact_dir / "meta.json").read_text())
        if meta.get("format_version") != SUPPORTED_FORMAT:
            raise ValueError(f"unsupported artifact format {meta.get('format_version')!r}")
        cand = pd.read_parquet(artifact_dir / "candidates.parquet")
        cand = cand.sort_values(["customer_id", "article_id"], kind="stable", ignore_index=True)
        self.meta = meta
        self.booster = lgb.Booster(model_file=str(artifact_dir / "model.txt"))
        self.features = cand[meta["features"]].to_numpy(dtype=np.float32)
        self.article_ids = cand["article_id"].to_numpy(dtype=np.int64)
        ids = cand["customer_id"].to_numpy()
        starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
        ends = np.r_[starts[1:], len(ids)]
        self.slices = {ids[s]: (int(s), int(e)) for s, e in zip(starts, ends, strict=True)}
        self.popular = [int(a) for a in json.loads((artifact_dir / "popularity.json").read_text())]

    def recommend(self, customer_id: str, k: int) -> Recommendation:
        span = self.slices.get(customer_id)
        if span is None:
            return Recommendation(
                customer_id=customer_id, source="popularity_fallback", article_ids=self.popular[:k]
            )
        lo, hi = span
        scores = self.booster.predict(self.features[lo:hi])
        arts = self.article_ids[lo:hi]
        order = np.lexsort((arts, -scores))  # best score first, ties by article id
        ids = [int(a) for a in arts[order][:k]]
        out_scores: list[float | None] = [float(s) for s in scores[order][:k]]
        for a in self.popular:  # a customer with fewer than k candidates is filled from popularity
            if len(ids) >= k:
                break
            if a not in ids:
                ids.append(a)
                out_scores.append(None)  # filled in, not scored by the ranker
        return Recommendation(
            customer_id=customer_id, source="ranker", article_ids=ids, scores=out_scores
        )


def create_app(artifact_dir: str | Path | None = None) -> FastAPI:
    directory = Path(artifact_dir or os.getenv("ARTIFACT_DIR", "artifacts"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.store = Store(directory)
        except (OSError, ValueError, KeyError, lgb.basic.LightGBMError):
            app.state.store = None  # /health reports not ready, /recommend answers 503
        yield

    app = FastAPI(title="hm-recsys", version="0.1.0", lifespan=lifespan)

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        store: Store | None = request.app.state.store
        if store is None:
            return Health(status="ok", ready=False)
        m = store.meta
        return Health(
            status="ok",
            ready=True,
            cutoff=m["cutoff"],
            customers=m["n_customers"],
            n_features=m["n_features"],
            built_utc=m["built_utc"],
        )

    @app.get("/recommend", response_model=Recommendation)
    def recommend(
        request: Request,
        customer_id: str = Query(..., min_length=1, max_length=128),
        k: int = Query(12, ge=1, le=MAX_K),
    ) -> Recommendation:
        store: Store | None = request.app.state.store
        if store is None:
            raise HTTPException(status_code=503, detail="model artifacts are not loaded")
        return store.recommend(customer_id, k)

    return app
