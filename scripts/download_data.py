"""Download the H&M competition files with the Kaggle CLI.

Prerequisites (done by you, not by this script):
  1. Accept the rules at https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations
  2. Put your Kaggle token in ~/.kaggle/kaggle.json (or set KAGGLE_USERNAME / KAGGLE_KEY in .env).
     Never commit it.
"""

import subprocess
import sys
import zipfile
from pathlib import Path

COMPETITION = "h-and-m-personalized-fashion-recommendations"
FILES = ["transactions_train.csv", "articles.csv", "customers.csv"]
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if (RAW / name).exists():
            print(f"{name}: already present")
            continue
        cmd = ["kaggle", "competitions", "download", "-c", COMPETITION, "-f", name, "-p", str(RAW)]
        print("running:", " ".join(cmd))
        if subprocess.run(cmd).returncode != 0:
            print("download failed: check the Kaggle CLI install, token and accepted rules")
            return 1
        archive = RAW / f"{name}.zip"
        if archive.exists():
            with zipfile.ZipFile(archive) as z:
                z.extractall(RAW)
            archive.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
