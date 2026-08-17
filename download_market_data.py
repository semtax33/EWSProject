"""Download signal-index OHLCV and investable adjusted prices by market."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.markets import MARKET_PROFILES, get_market_profile


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _one_ticker_frame(frame, ticker):
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame
    if ticker in frame.columns.get_level_values(-1):
        return frame.xs(ticker, axis=1, level=-1)
    frame = frame.copy()
    frame.columns = frame.columns.get_level_values(0)
    return frame


def _write_csv_atomic(frame, output, *, index):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temporary, index=index)
    temporary.replace(output)


def download_signal_index(profile, *, start, end, output, metadata_output):
    data = yf.download(
        profile.ticker,
        start=start,
        end=end,
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="column",
    )
    data = _one_ticker_frame(data, profile.ticker)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = set(required).difference(data.columns)
    if data.empty or missing:
        raise RuntimeError(
            f"{profile.ticker} OHLCV download failed; missing={sorted(missing)}"
        )
    clean = data[required].copy()
    clean.index = pd.to_datetime(clean.index)
    clean = clean[~clean.index.duplicated(keep="last")].sort_index()
    clean = clean.dropna(subset=["Open", "High", "Low", "Close"])
    if len(clean) < 1200 or ((clean["High"] < clean["Low"]) | (clean["Close"] <= 0)).any():
        raise RuntimeError(f"{profile.ticker} OHLCV history failed validation")
    clean.index.name = "Date"
    _write_csv_atomic(clean, output, index=True)
    metadata = {
        "schema_version": 1,
        "market": profile.display_name,
        "ticker": profile.ticker,
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
        "role": "signal target and price-index benchmark proxy",
    }
    Path(metadata_output).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return clean


def download_investable(profile, *, start, end, output):
    data = yf.download(
        profile.investable_ticker,
        start=start,
        end=end,
        auto_adjust=False,
        actions=True,
        progress=False,
        group_by="column",
    )
    data = _one_ticker_frame(data, profile.investable_ticker)
    if data.empty or "Adj Close" not in data.columns:
        raise RuntimeError(
            f"{profile.investable_ticker} adjusted-price download failed"
        )
    adjusted = data["Adj Close"].rename("adjusted_close").dropna()
    frame = adjusted.reset_index()
    frame.columns = ["date", "adjusted_close"]
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    if len(frame) < 1200 or (frame["adjusted_close"] <= 0).any():
        raise RuntimeError(
            f"{profile.investable_ticker} adjusted-price history failed validation"
        )
    _write_csv_atomic(frame, output, index=False)
    metadata = {
        "schema_version": 1,
        "market": profile.display_name,
        "instrument": profile.investable_instrument,
        "ticker": profile.investable_ticker,
        "source": "Yahoo Finance via yfinance",
        "value_semantics": "adjusted close including distributions and splits",
        "first_date": frame["date"].iloc[0],
        "last_date": frame["date"].iloc[-1],
        "observations": len(frame),
        "downloaded_at": datetime.now().astimezone().isoformat(),
        "sha256": _sha256(output),
        "portfolio_execution": "signal at completed month end; execute next month",
    }
    Path(output).with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return frame


def download_market_data(
    market,
    *,
    end=None,
    index_start=None,
    investable_start=None,
    signal_output=None,
    investable_output=None,
):
    profile = get_market_profile(market)
    end = end or (date.today() + timedelta(days=1)).isoformat()
    signal_output = Path(signal_output or profile.signal_file)
    investable_output = Path(investable_output or profile.investable_file)
    signal = download_signal_index(
        profile,
        start=index_start or profile.index_start,
        end=end,
        output=signal_output,
        metadata_output=signal_output.with_suffix(".metadata.json"),
    )
    investable = download_investable(
        profile,
        start=investable_start or profile.investable_start,
        end=end,
        output=investable_output,
    )
    print(
        f"{profile.display_name}: index {len(signal):,} rows -> {signal_output}; "
        f"{profile.investable_instrument} {len(investable):,} rows -> {investable_output}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download EWS market target and investable return data"
    )
    parser.add_argument("--market", choices=MARKET_PROFILES, default="kospi200")
    parser.add_argument("--end", help="exclusive YYYY-MM-DD end date")
    parser.add_argument("--index-start")
    parser.add_argument("--investable-start")
    parser.add_argument("--signal-output")
    parser.add_argument("--investable-output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_market_data(
        args.market,
        end=args.end,
        index_start=args.index_start,
        investable_start=args.investable_start,
        signal_output=args.signal_output,
        investable_output=args.investable_output,
    )
