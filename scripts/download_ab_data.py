"""Download the public Cookie Cats A/B test data with the Kaggle CLI into data/raw/cookie_cats/.

Same token setup as download_data.py. The data is not committed.
"""

import subprocess
import sys
import zipfile
from pathlib import Path

DATASET = "mursideyarkin/mobile-games-ab-testing-cookie-cats"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "cookie_cats"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "cookie_cats.csv").exists():
        print("cookie_cats.csv: already present")
        return 0
    cmd = ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(OUT)]
    print("running:", " ".join(cmd))
    if subprocess.run(cmd).returncode != 0:
        print("download failed: check the Kaggle CLI install and token")
        return 1
    for archive in OUT.glob("*.zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(OUT)
        archive.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
