"""Exploratory numbers for the README, computed from data/processed/transactions.parquet.

Writes results/eda.json and prints it. Nothing here is hard-coded: paste the output into the README.
"""

import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
PQ = (ROOT / "data" / "processed" / "transactions.parquet").as_posix()


def main() -> None:
    con = duckdb.connect()
    q = lambda sql: con.execute(sql).fetchall()  # noqa: E731
    src = f"read_parquet('{PQ}')"

    rows, customers, articles = q(
        f"SELECT count(*), count(DISTINCT customer_id), count(DISTINCT article_id) FROM {src}"
    )[0]

    per_customer = q(
        f"""SELECT quantile_cont(n, [0.25, 0.5, 0.75, 0.9]), avg(n) FROM
            (SELECT count(*) AS n FROM {src} GROUP BY customer_id)"""
    )[0]

    # Repeat purchase: share of purchases where the customer bought the same article earlier.
    repeat = q(
        f"""SELECT avg(CASE WHEN prev IS NOT NULL THEN 1.0 ELSE 0.0 END) FROM (
              SELECT lag(t_dat) OVER (PARTITION BY customer_id, article_id ORDER BY t_dat) AS prev
              FROM {src})"""
    )[0][0]

    weekly = q(
        f"""SELECT date_trunc('week', t_dat)::VARCHAR AS week, count(*) AS purchases
            FROM {src} GROUP BY 1 ORDER BY 1"""
    )

    # Long tail: share of purchases covered by the top 1% / 10% of articles.
    tail = q(
        f"""WITH c AS (SELECT count(*) AS n FROM {src} GROUP BY article_id),
                 r AS (SELECT n, row_number() OVER (ORDER BY n DESC) AS rk,
                      count(*) OVER () AS total FROM c)
            SELECT sum(CASE WHEN rk <= total * 0.01 THEN n END) / sum(n),
                   sum(CASE WHEN rk <= total * 0.10 THEN n END) / sum(n) FROM r"""
    )[0]

    out = {
        "rows": rows,
        "customers": customers,
        "articles": articles,
        "purchases_per_customer_quantiles_25_50_75_90": per_customer[0],
        "purchases_per_customer_mean": per_customer[1],
        "repeat_purchase_rate": repeat,
        "weekly_purchases": weekly,
        "top_1pct_articles_share_of_purchases": tail[0],
        "top_10pct_articles_share_of_purchases": tail[1],
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "eda.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
