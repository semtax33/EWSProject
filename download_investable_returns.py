"""Download a reproducible distribution-adjusted KODEX 200 price series."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf


TICKER = "069500.KS"
OUTPUT = Path("Data") / "MARKET" / "KODEX200_adjusted.csv"


def main(end=None):
    end = end or (pd.Timestamp.today().normalize() + pd.Timedelta(days=1))
    history = yf.download(
        TICKER,
        start="2002-01-01",
        end=pd.Timestamp(end).date().isoformat(),
        auto_adjust=False,
        actions=True,
        progress=False,
    )
    if history.empty:
        raise RuntimeError(f"No adjusted-price data downloaded for {TICKER}")
    adjusted = history["Adj Close"]
    if isinstance(adjusted, pd.DataFrame):
        adjusted = adjusted[TICKER]
    frame = adjusted.rename("adjusted_close").dropna().reset_index()
    frame.columns = ["date", "adjusted_close"]
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    if len(frame) < 1200 or (frame["adjusted_close"] <= 0).any():
        raise RuntimeError("KODEX 200 adjusted-price history failed validation")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(OUTPUT)
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    metadata = {
        "schema_version": 1,
        "instrument": "KODEX 200 ETF",
        "ticker": TICKER,
        "source": "Yahoo Finance via yfinance",
        "value_semantics": "adjusted close including distributions and splits",
        "first_date": frame["date"].iloc[0],
        "last_date": frame["date"].iloc[-1],
        "observations": len(frame),
        "downloaded_at": datetime.now().astimezone().isoformat(),
        "sha256": digest,
        "portfolio_execution": "signal at completed month end; execute next month",
    }
    OUTPUT.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"{TICKER}: {len(frame):,} rows, {frame['date'].iloc[0]} ~ "
        f"{frame['date'].iloc[-1]} -> {OUTPUT} ({digest[:12]})"
    )


if __name__ == "__main__":
    main()
