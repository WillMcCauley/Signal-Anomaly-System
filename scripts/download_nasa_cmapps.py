from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


DEFAULT_URL = "https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NASA C-MAPSS turbofan degradation dataset.")
    parser.add_argument("--url", default=DEFAULT_URL, help="NASA data portal zip URL.")
    parser.add_argument("--output-dir", default="data/raw", help="Directory for raw NASA files.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "nasa_cmapps.zip"

    print(f"Downloading NASA C-MAPSS data to {zip_path} ...")
    urlretrieve(args.url, zip_path)

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)
    print(f"Done. Raw files are available in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
