"""Download the expanded FRED raw-series universe declared in the catalog."""

import argparse
from pathlib import Path

from src.config import FRED_DIR, RAW_SERIES_CATALOG_FILE
from src.raw_catalog import load_raw_series_catalog
from src.raw_download import download_catalog


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default=str(RAW_SERIES_CATALOG_FILE))
    parser.add_argument("--output-dir", default=str(FRED_DIR))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main(catalog_path, output_dir, refresh=False, workers=8):
    catalog = load_raw_series_catalog(catalog_path)
    report = download_catalog(
        catalog, output_dir, refresh=refresh, workers=workers
    )
    report_path = Path(output_dir).parent / "DERIVED" / "raw_series_download_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)
    counts = report["status"].value_counts().to_dict()
    print(f"Download report: {report_path}")
    print(f"Status counts: {counts}")
    if report["status"].eq("failed").any():
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    args = parse_args()
    main(args.catalog, args.output_dir, refresh=args.refresh, workers=args.workers)
