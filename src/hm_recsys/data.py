"""Loading, subsampling and time-based splitting of the H&M transactions with DuckDB."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd

WEEK = timedelta(days=7)


@dataclass(frozen=True)
class Split:
    """Half-open date windows. Train is everything before `val_start`; validation is the 7 days
    after it; test is the last 7 days of the data and is touched once at the end."""

    val_start: date
    test_start: date
    test_end: date  # exclusive

    @property
    def val_end(self) -> date:
        return self.test_start


def make_split(last_date: date) -> Split:
    """Test = the last 7 days ending on `last_date`, validation = the 7 days before that."""
    test_end = last_date + timedelta(days=1)
    test_start = test_end - WEEK
    return Split(val_start=test_start - WEEK, test_start=test_start, test_end=test_end)


def prepare_transactions(
    csv_path: Path,
    out_path: Path,
    weeks: int = 16,
    customer_fraction: float = 1.0,
) -> dict:
    """Read the raw CSV, keep the last `weeks` weeks and an optional deterministic customer
    sample, write parquet. Returns the sample description to record with every result.

    The customer sample is a hash of customer_id, so the same customers are kept on every run and
    in every week (a customer is never half in the sample).
    """
    if not 0 < customer_fraction <= 1:
        raise ValueError("customer_fraction must be in (0, 1]")
    buckets = 10_000
    keep = max(1, round(customer_fraction * buckets))
    con = duckdb.connect()
    src = f"read_csv('{Path(csv_path).as_posix()}', header=true, dateformat='%Y-%m-%d')"
    last = con.execute(f"SELECT max(t_dat) FROM {src}").fetchone()[0]
    first = last - timedelta(days=7 * weeks - 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY (
            SELECT t_dat::DATE AS t_dat, customer_id, article_id::BIGINT AS article_id,
                   price::DOUBLE AS price, sales_channel_id::TINYINT AS sales_channel_id
            FROM {src}
            WHERE t_dat >= DATE '{first}'
              AND hash(customer_id) % {buckets} < {keep}
        ) TO '{Path(out_path).as_posix()}' (FORMAT parquet)
        """
    )
    n_rows, n_customers, n_articles = con.execute(
        f"SELECT count(*), count(DISTINCT customer_id), count(DISTINCT article_id) "
        f"FROM read_parquet('{Path(out_path).as_posix()}')"
    ).fetchone()
    return {
        "first_date": str(first),
        "last_date": str(last),
        "weeks": weeks,
        "customer_fraction": keep / buckets,
        "rows": n_rows,
        "customers": n_customers,
        "articles": n_articles,
    }


def prepare_customers(customers_csv: Path, transactions_parquet: Path, out_path: Path) -> int:
    """Write customer_id and age for the customers in the transaction sample. Returns the row
    count. Age is NULL where H&M did not record it."""
    con = duckdb.connect()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY (
            SELECT customer_id, age::DOUBLE AS age
            FROM read_csv('{Path(customers_csv).as_posix()}', header=true)
            WHERE customer_id IN (
                SELECT DISTINCT customer_id
                FROM read_parquet('{Path(transactions_parquet).as_posix()}')
            )
        ) TO '{Path(out_path).as_posix()}' (FORMAT parquet)
        """
    )
    return con.execute(
        f"SELECT count(*) FROM read_parquet('{Path(out_path).as_posix()}')"
    ).fetchone()[0]


def prepare_articles(articles_csv: Path, out_path: Path) -> int:
    """Write the article attributes used as features (category codes only). Returns the rows."""
    con = duckdb.connect()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY (
            SELECT article_id::BIGINT AS article_id, product_type_no, product_group_name,
                   graphical_appearance_no, colour_group_code, department_no, index_group_no
            FROM read_csv('{Path(articles_csv).as_posix()}', header=true)
        ) TO '{Path(out_path).as_posix()}' (FORMAT parquet)
        """
    )
    return con.execute(
        f"SELECT count(*) FROM read_parquet('{Path(out_path).as_posix()}')"
    ).fetchone()[0]


def split_transactions(df: pd.DataFrame, split: Split) -> dict[str, pd.DataFrame]:
    """Return train / val / test frames from a transactions frame with a `t_dat` date column."""
    d = pd.to_datetime(df["t_dat"]).dt.date
    return {
        "train": df[d < split.val_start],
        "val": df[(d >= split.val_start) & (d < split.val_end)],
        "test": df[(d >= split.test_start) & (d < split.test_end)],
    }


def history_before(df: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    """The only rows a feature for a prediction window starting at `cutoff` may use."""
    return df[pd.to_datetime(df["t_dat"]).dt.date < cutoff]


def actuals_by_customer(df: pd.DataFrame) -> dict[str, set]:
    """Map customer_id -> set of purchased article_ids (the ground truth for evaluation)."""
    return df.groupby("customer_id")["article_id"].agg(set).to_dict()
