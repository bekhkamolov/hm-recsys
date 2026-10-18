"""FastAPI service: rank precomputed candidates with the LightGBM ranker, fall back to popularity.

Runtime only: this module needs lightgbm, numpy, pandas and fastapi, and imports none of the
training code (a test checks that). Artifacts are built by `hm_recsys.artifacts`.
Run with: uvicorn hm_recsys.api:create_app --factory   (artifact folder from $ARTIFACT_DIR)

  GET /recommend?customer_id=...&k=12   top-k article ids for a customer
  GET /health                           readiness and what snapshot is loaded
  GET /stats                            fallback share and scoring time since start-up
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
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


class StatsOut(BaseModel):
    requests: int
    ranker_requests: int
    fallback_requests: int
    fallback_share: float | None  # None before the first request
    scoring_ms_mean: float | None  # time inside the model, last `window` requests
    scoring_ms_p95: float | None
    window: int


class RequestStats:
    """In-process counters since start-up plus a rolling window of scoring times. No customer ids
    are kept. With several workers each process has its own counters."""

    def __init__(self, window: int = 1000):
        self.window = window
        self._lock = threading.Lock()
        self._times: deque[float] = deque(maxlen=window)
        self._counts = {"ranker": 0, "popularity_fallback": 0}

    def record(self, source: str, seconds: float) -> None:
        with self._lock:
            self._counts[source] += 1
            self._times.append(seconds * 1000)

    def snapshot(self) -> StatsOut:
        with self._lock:
            times = sorted(self._times)
            ranker, fallback = self._counts["ranker"], self._counts["popularity_fallback"]
        total = ranker + fallback
        return StatsOut(
            requests=total,
            ranker_requests=ranker,
            fallback_requests=fallback,
            fallback_share=fallback / total if total else None,
            scoring_ms_mean=sum(times) / len(times) if times else None,
            scoring_ms_p95=times[min(len(times) - 1, int(0.95 * len(times)))] if times else None,
            window=self.window,
        )


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>hm-recsys</title>
<style>
  :root {{ --bg:#fff; --fg:#1a1a1a; --muted:#5b6470; --line:#d9dde3; --link:#0b5fff; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#14171c; --fg:#e8eaed; --muted:#a0a8b3; --line:#2b313a; --link:#7aa7ff; }}
  }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
         font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif; }}
  main {{ max-width:42rem; margin:0 auto; padding:2rem 16px 3rem; }}
  h1 {{ font-size:1.6rem; margin:0 0 .25rem; }}
  p.sub {{ color:var(--muted); margin:0 0 1.5rem; }}
  a {{ color:var(--link); }}
  ul {{ padding-left:1.2rem; }} li {{ margin:.35rem 0; }}
  .box {{ border:1px solid var(--line); border-radius:8px; padding:.75rem 1rem; margin:1.25rem 0; }}
  .box dt {{ color:var(--muted); float:left; clear:left; width:9rem; }}
  .box dd {{ margin:0 0 .25rem 9rem; }}
  code {{ font-family:ui-monospace,Consolas,monospace; font-size:.92em; overflow-wrap:anywhere; }}
</style></head>
<body><main>
<h1>hm-recsys</h1>
<p class="sub">Product recommendations for a fashion retailer: precomputed candidates ranked by a
LightGBM model, with a popularity fallback for unknown customers.</p>
{model}
<h2>Try it</h2>
<ul>
  <li><a href="/recommend?customer_id={example}">/recommend?customer_id={example}</a>
      &ndash; top 12 for a known customer (ranked by the model)</li>
  <li><a href="/recommend?customer_id=someone-new">/recommend?customer_id=someone-new</a>
      &ndash; unknown customer, popularity fallback</li>
  <li><a href="/docs">/docs</a> &ndash; interactive API documentation</li>
  <li><a href="/health">/health</a> and <a href="/stats">/stats</a> &ndash; readiness and
      request statistics</li>
</ul>
<p class="sub">Add <code>&amp;k=5</code> to ask for fewer items (1 to 50).</p>
</main></body></html>
"""


def landing_page(store: Store | None) -> str:
    """Small HTML page for the root URL. Everything shown from the model is escaped."""
    if store is None:
        model = (
            '<div class="box">Model artifacts are not loaded, '
            "so recommendations are unavailable.</div>"
        )
        example = "c0001"
    else:
        m = store.meta
        rows = [
            ("Snapshot date", m["cutoff"]),
            ("Data", m.get("data", "not stated")),
            ("Customers in table", f"{m['n_customers']:,}"),
            ("Features", str(m["n_features"])),
        ]
        body = "".join(f"<dt>{escape(k)}</dt><dd>{escape(str(v))}</dd>" for k, v in rows)
        model = f'<dl class="box">{body}</dl>'
        example = (m.get("example_customer_ids") or ["c0001"])[0]
    return _PAGE.format(model=model, example=escape(quote(example), quote=True))


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
        app.state.stats = RequestStats()
        try:
            app.state.store = Store(directory)
        except (OSError, ValueError, KeyError, lgb.basic.LightGBMError):
            app.state.store = None  # /health reports not ready, /recommend answers 503
        yield

    app = FastAPI(title="hm-recsys", version="0.1.0", lifespan=lifespan)

    @app.get("/", include_in_schema=False, response_class=HTMLResponse)
    def index(request: Request) -> str:
        return landing_page(request.app.state.store)

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
        start = time.perf_counter()
        result = store.recommend(customer_id, k)
        request.app.state.stats.record(result.source, time.perf_counter() - start)
        return result

    @app.get("/stats", response_model=StatsOut)
    def stats(request: Request) -> StatsOut:
        """Operational numbers for monitoring: how often the fallback is used and scoring time."""
        return request.app.state.stats.snapshot()

    return app
