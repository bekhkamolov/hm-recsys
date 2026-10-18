from datetime import date, timedelta

import pandas as pd
import pytest


@pytest.fixture
def transactions() -> pd.DataFrame:
    """Synthetic transactions: 6 customers, 5 articles, 5 weeks starting 2020-08-03 (a Monday)."""
    start = date(2020, 8, 3)
    rows = []
    for day in range(35):
        d = start + timedelta(days=day)
        for c in range(6):
            if (day + c) % 3 == 0:
                rows.append((d, f"c{c}", 100 + (day + c) % 5, 10.0 + c, 1 + c % 2))
    return pd.DataFrame(
        rows, columns=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
    )
