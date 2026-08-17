"""Download a clean, reproducible KOSPI200 (^KS200) OHLCV file."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import KOSPI200_MARKET_FILE, KOSPI200_MARKET_METADATA_FILE


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def download_kospi200(start, end, output, metadata_output=None):
    ticker = "^KS200"
    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="column",
    )
    if data.empty:
        raise RuntimeError("Yahoo Finance returned no KOSPI200 observations")

    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(-1):
            data = data.xs(ticker, axis=1, level=-1)
        else:
            data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = set(required).difference(data.columns)
    if missing:
        raise RuntimeError(f"Downloaded OHLCV schema missing: {sorted(missing)}")

    clean = data[required].copy()
    clean.index = pd.to_datetime(clean.index)
    clean = clean[~clean.index.duplicated(keep="last")].sort_index()
    clean = clean.dropna(subset=["Close", "High", "Low", "Open"])
    if not clean.index.is_monotonic_increasing or clean.empty:
        raise RuntimeError("Downloaded KOSPI200 dates are invalid")
    if ((clean["High"] < clean["Low"]) | (clean["Close"] <= 0)).any():
        raise RuntimeError("Downloaded KOSPI200 prices failed range checks")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    clean.index.name = "Date"
    clean.to_csv(output)
    metadata_output = Path(metadata_output or KOSPI200_MARKET_METADATA_FILE)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(
        json.dumps(
            {
                "ticker": ticker,
                "provider": "Yahoo Finance via yfinance",
                "downloaded_at": datetime.now().astimezone().isoformat(),
                "requested_start": start,
                "requested_end_exclusive": end,
                "rows": len(clean),
                "first_date": clean.index.min().date().isoformat(),
                "last_date": clean.index.max().date().isoformat(),
                "sha256": _sha256(output),
                "schema": required,
                "schema_validated": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Saved {len(clean):,} KOSPI200 rows "
        f"({clean.index.min().date()} ~ {clean.index.max().date()}) to {output}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="1996-01-01")
    parser.add_argument(
        "--end",
        default=(date.today() + timedelta(days=1)).isoformat(),
        help="exclusive end date",
    )
    parser.add_argument("--output", default=str(KOSPI200_MARKET_FILE))
    parser.add_argument("--metadata-output", default=str(KOSPI200_MARKET_METADATA_FILE))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_kospi200(args.start, args.end, args.output, args.metadata_output)
