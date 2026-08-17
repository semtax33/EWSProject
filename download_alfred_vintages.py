"""Download deployment-relevant series as month-end ALFRED vintages."""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from src.config import ALFRED_DIR, RAW_SERIES_CATALOG_FILE
from src.raw_catalog import load_raw_series_catalog
from src.vintage import download_monthly_vintage_series


def main(start="1988-01-31", end=None, workers=1, request_interval=0.35):
    if end is None:
        end = (pd.Timestamp(date.today()).to_period("M") - 1).to_timestamp("M")
    # Yield-curve factors and the MLP candidate's macro/FX sources are model
    # inputs; DGS3MO is the portfolio cash leg.  Monthly/quarterly series use
    # the latest reference-period value visible in each month-end vintage.
    series_aggregations = {
        "T10Y2Y": "mean",
        "T10Y3M": "mean",
        "DGS3MO": "last",
        "NFCPATAX": "latest_available",
        "RSAFS": "latest_available",
        "DEXUSAL": "mean",
    }
    catalog = load_raw_series_catalog(RAW_SERIES_CATALOG_FILE)
    valuation_sources = catalog.loc[
        catalog["source"].eq("FRED")
        & catalog["group"].eq("earnings_valuation")
        & catalog["enabled"]
    ]
    for row in valuation_sources.itertuples(index=False):
        aggregation = (
            "latest_available"
            if row.freq in {"monthly", "quarterly"}
            else row.agg
        )
        series_aggregations.setdefault(row.series_id, aggregation)
    for series_id, aggregation in series_aggregations.items():
        path = ALFRED_DIR / f"{series_id}_point_in_time.csv"
        frame, metadata = download_monthly_vintage_series(
            series_id,
            path,
            start=start,
            end=end,
            agg=aggregation,
            workers=workers,
            request_interval_seconds=request_interval,
        )
        print(
            f"{series_id}: {frame['value'].notna().sum():,}/{len(frame):,} "
            f"months -> {path} ({metadata['sha256'][:12]})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="1988-01-31")
    parser.add_argument("--end")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--request-interval", type=float, default=0.35)
    args = parser.parse_args()
    main(
        start=args.start,
        end=args.end,
        workers=args.workers,
        request_interval=args.request_interval,
    )
