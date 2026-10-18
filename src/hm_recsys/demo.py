"""Small synthetic data with some purchase structure, for tests and the CI smoke test.

Customers have favourite articles they buy again, plus a pull towards currently popular articles,
so a ranker trained on it has something to learn. It is not meant to resemble H&M statistically.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

START = date(2020, 7, 6)  # a Monday


def synthetic_transactions(
    n_customers: int = 200, n_articles: int = 60, weeks: int = 10, seed: int = 0
) -> pd.DataFrame:
    """Transactions with columns t_dat, customer_id, article_id, price, sales_channel_id."""
    rng = np.random.default_rng(seed)
    base_price = rng.uniform(0.01, 0.1, n_articles)
    popularity = rng.dirichlet(np.full(n_articles, 0.5))
    rows = []
    for c in range(n_customers):
        favourites = rng.choice(n_articles, size=rng.integers(3, 9), replace=False)
        rate = rng.uniform(0.02, 0.12)  # purchases per day
        channel = int(rng.integers(1, 3))
        for day in range(weeks * 7):
            if rng.random() < rate:
                pool = favourites if rng.random() < 0.6 else np.arange(n_articles)
                weights = popularity[pool] / popularity[pool].sum()
                a = int(rng.choice(pool, p=weights))
                price = float(base_price[a] * rng.uniform(0.8, 1.0))
                rows.append((START + timedelta(days=day), f"c{c:04d}", 1000 + a, price, channel))
    return pd.DataFrame(
        rows, columns=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
    )


def synthetic_meta(
    transactions: pd.DataFrame, n_articles: int = 60, seed: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(customers with age, articles with category columns) matching `synthetic_transactions`."""
    rng = np.random.default_rng(seed)
    ids = sorted(transactions["customer_id"].unique())
    customers = pd.DataFrame(
        {"customer_id": ids, "age": rng.integers(16, 75, len(ids)).astype(float)}
    )
    customers.loc[rng.random(len(ids)) < 0.03, "age"] = np.nan
    articles = pd.DataFrame(
        {
            "article_id": 1000 + np.arange(n_articles),
            "product_type_no": rng.integers(250, 270, n_articles),
            "product_group_name": rng.choice(
                ["Upper body", "Lower body", "Accessories"], n_articles
            ),
            "graphical_appearance_no": rng.integers(1010000, 1010010, n_articles),
            "colour_group_code": rng.integers(1, 12, n_articles),
            "department_no": rng.integers(1200, 1230, n_articles),
            "index_group_no": rng.integers(1, 4, n_articles),
        }
    )
    return customers, articles
