"""Smoke-test a running hm-recsys service (local, Docker or the Render deployment).

    python scripts/smoke_test.py https://<your-service>.onrender.com [--customer c0001]

Exits non-zero if /health is not ready, a known customer is not ranked by the model, an unknown
customer does not get the popularity fallback, or /stats does not count the requests. The first
request to a free Render service can take a minute while it wakes up; the script waits.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def get(url: str, timeout: float = 90) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("base_url")
    p.add_argument("--customer", default="c0001", help="a customer id present in the artifacts")
    args = p.parse_args()
    base = args.base_url.rstrip("/")

    status, health = 0, {}
    for attempt in range(1, 11):  # a sleeping free instance needs a while to start
        try:
            status, health = get(f"{base}/health")
            if status == 200:
                break
        except OSError as e:
            print(f"attempt {attempt}: {e}")
        time.sleep(10)
    print("health:", health)
    ok = status == 200 and health.get("ready") is True
    if not ok:
        print("FAIL: service is not ready")
        return 1

    before = get(f"{base}/stats")[1]
    t0 = time.perf_counter()
    _, known = get(f"{base}/recommend?customer_id={args.customer}&k=12")
    elapsed = (time.perf_counter() - t0) * 1000
    _, unknown = get(f"{base}/recommend?customer_id=does-not-exist&k=12")
    bad_status, _ = get(f"{base}/recommend")
    after = get(f"{base}/stats")[1]

    checks = {
        "known customer ranked by the model": known.get("source") == "ranker"
        and len(known["article_ids"]) == 12,
        "unknown customer gets the popularity fallback": unknown.get("source")
        == "popularity_fallback",
        "missing customer_id returns 422": bad_status == 422,
        "/stats counted the two recommend calls": after["requests"] - before["requests"] == 2,
    }
    for name, passed in checks.items():
        print(("PASS" if passed else "FAIL"), name)
    print(f"known-customer request took {elapsed:.0f} ms (includes network)")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
