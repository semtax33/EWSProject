"""Point-in-time KOSPI top-200 price-return proxy from the marcap panel."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
SOURCE_COLUMNS = [
    "Code",
    "Name",
    "ChangesRatio",
    "Marcap",
    "Market",
    "observation_date",
]


def _sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _is_common_stock(code):
    """KRX common-share issue codes conventionally end in zero."""
    return code.astype("string").str.fullmatch(r"\d{5}0", na=False)


def _top_constituents(day, count):
    eligible = day.loc[
        day["Marcap"].gt(0)
        & day["ChangesRatio"].notna()
        & _is_common_stock(day["Code"])
    ].copy()
    eligible = eligible.sort_values(
        ["Marcap", "Code"], ascending=[False, True], kind="stable"
    ).drop_duplicates("Code", keep="last")
    selected = eligible.head(count)
    total = selected["Marcap"].sum()
    if len(selected) < count or not np.isfinite(total) or total <= 0:
        return None
    return pd.Series(
        selected["Marcap"].to_numpy(dtype=float) / float(total),
        index=selected["Code"].astype(str),
        dtype=float,
    )


def _completed_month_end(source_last_date, asof_date=None):
    asof = pd.Timestamp(asof_date or datetime.now().astimezone()).tz_localize(None)
    source_last = pd.Timestamp(source_last_date).tz_localize(None)
    if source_last.to_period("M") >= asof.to_period("M"):
        return (asof.to_period("M") - 1).to_timestamp("M")
    return source_last.to_period("M").to_timestamp("M")


def _process_day(
    day,
    *,
    constituent_count,
    current_month,
    holdings,
    month_end_candidate,
):
    date = pd.Timestamp(day["observation_date"].iloc[0])
    month = date.to_period("M")
    if current_month is None:
        current_month = month
    elif month != current_month:
        holdings = month_end_candidate
        current_month = month

    daily_return = np.nan
    return_coverage = np.nan
    if holdings is not None:
        returns = (
            day.drop_duplicates("Code", keep="last")
            .set_index(day["Code"].astype(str))["ChangesRatio"]
            .astype(float)
            .div(100.0)
            .replace([np.inf, -np.inf], np.nan)
        )
        aligned = returns.reindex(holdings.index)
        observed = aligned.notna()
        return_coverage = float(holdings.loc[observed].sum())
        effective_returns = aligned.fillna(0.0)
        daily_return = float((holdings * effective_returns).sum())
        gross = 1.0 + daily_return
        if gross <= 0:
            raise ValueError(f"Marcap top-200 proxy lost 100% on {date.date()}")
        holdings = holdings * (1.0 + effective_returns) / gross

    month_end_candidate = _top_constituents(day, constituent_count)
    row = {
        "date": date,
        "return": daily_return,
        "return_coverage": return_coverage,
        "constituents": (
            int(len(holdings)) if holdings is not None else 0
        ),
    }
    return row, current_month, holdings, month_end_candidate


def build_marcap_kospi200_proxy(
    source_path,
    output_path,
    metadata_path,
    *,
    constituent_count=200,
    chunksize=500_000,
    asof_date=None,
):
    """Build a monthly rebalanced, prior-month-end top-200 KOSPI proxy.

    Official historical KOSPI200 membership is not present in marcap.  This
    series therefore selects the 200 largest KOSPI common shares at each
    month-end, applies those weights from the next trading month, and lets the
    weights drift between month-end rebalances.
    """
    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    metadata_path = Path(metadata_path).resolve()
    source_stat = source_path.stat()

    reader = pd.read_csv(
        source_path,
        usecols=SOURCE_COLUMNS,
        dtype={
            "Code": "string",
            "Name": "string",
            "ChangesRatio": "float64",
            "Marcap": "float64",
            "Market": "string",
        },
        parse_dates=["observation_date"],
        chunksize=chunksize,
    )

    pending = pd.DataFrame(columns=SOURCE_COLUMNS)
    rows = []
    source_rows = 0
    first_date = None
    last_date = None
    current_month = None
    holdings = None
    month_end_candidate = None

    for chunk_number, chunk in enumerate(reader, start=1):
        chunk = chunk.loc[
            chunk["Market"].eq("KOSPI")
            & chunk["Code"].notna()
            & chunk["observation_date"].notna()
        ].copy()
        source_rows += len(chunk)
        if not pending.empty:
            chunk = pd.concat([pending, chunk], ignore_index=True)
        chunk = chunk.sort_values(["observation_date", "Code"], kind="stable")
        if chunk.empty:
            continue

        final_date = chunk["observation_date"].max()
        completed = chunk.loc[chunk["observation_date"].lt(final_date)]
        pending = chunk.loc[chunk["observation_date"].eq(final_date)].copy()
        for date, day in completed.groupby("observation_date", sort=True):
            row, current_month, holdings, month_end_candidate = _process_day(
                day,
                constituent_count=constituent_count,
                current_month=current_month,
                holdings=holdings,
                month_end_candidate=month_end_candidate,
            )
            rows.append(row)
            first_date = date if first_date is None else min(first_date, date)
            last_date = date if last_date is None else max(last_date, date)
        print(f"marcap top-200 chunk {chunk_number:,}: {source_rows:,} KOSPI rows")

    if not pending.empty:
        date = pending["observation_date"].iloc[0]
        row, current_month, holdings, month_end_candidate = _process_day(
            pending,
            constituent_count=constituent_count,
            current_month=current_month,
            holdings=holdings,
            month_end_candidate=month_end_candidate,
        )
        rows.append(row)
        first_date = date if first_date is None else min(first_date, date)
        last_date = date if last_date is None else max(last_date, date)

    daily = pd.DataFrame(rows).set_index("date").sort_index()
    complete_through = _completed_month_end(last_date, asof_date=asof_date)
    daily = daily.loc[daily.index <= complete_through].copy()
    daily = daily.loc[daily["return"].notna()].copy()
    daily["close"] = (1.0 + daily["return"]).cumprod() * 100.0
    monthly = daily.resample("ME").agg(
        close=("close", "last"),
        return_coverage=("return_coverage", "min"),
        constituents=("constituents", "min"),
    )
    monthly = monthly.dropna(subset=["close"])
    monthly.index.name = "date"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_path)
    source_hash = _sha256(source_path)
    output_hash = _sha256(output_path)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "built_at": datetime.now().astimezone().isoformat(),
        "source": str(source_path),
        "source_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "source_sha256": source_hash,
        "source_rows": source_rows,
        "source_first_date": str(pd.Timestamp(first_date).date()),
        "source_last_date": str(pd.Timestamp(last_date).date()),
        "complete_through": str(complete_through.date()),
        "output": str(output_path),
        "output_sha256": output_hash,
        "output_rows": len(monthly),
        "value_column": "close",
        "return_type": "price_return",
        "universe": "KOSPI common shares in the supplied marcap panel",
        "constituent_count": constituent_count,
        "selection_rule": (
            "largest market capitalization at each month-end; applied next month"
        ),
        "weighting_rule": "month-end market-cap weights drift daily until rebalance",
        "official_kospi200_membership_available": False,
        "is_official_kospi200": False,
        "proxy_name": "Marcap KOSPI Top-200 Price Proxy",
        "minimum_daily_weight_coverage": float(monthly["return_coverage"].min()),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return monthly, metadata


def load_marcap_kospi200_proxy(source_path, output_path, metadata_path):
    """Load a proxy cache only when it matches the current source panel."""
    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    metadata_path = Path(metadata_path).resolve()
    if not output_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Marcap KOSPI top-200 proxy cache is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_stat = source_path.stat()
    if (
        metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("source_bytes") != source_stat.st_size
        or metadata.get("source_mtime_ns") != source_stat.st_mtime_ns
    ):
        raise RuntimeError("Marcap KOSPI top-200 proxy cache is stale")
    if metadata.get("output_sha256") != _sha256(output_path):
        raise RuntimeError("Marcap KOSPI top-200 proxy cache hash mismatch")
    frame = pd.read_csv(output_path, index_col="date", parse_dates=True)
    required = {"close", "return_coverage", "constituents"}
    if required.difference(frame.columns):
        raise RuntimeError("Marcap KOSPI top-200 proxy cache schema mismatch")
    return frame.sort_index(), metadata


def ensure_marcap_kospi200_proxy(source_path, output_path, metadata_path):
    try:
        return load_marcap_kospi200_proxy(source_path, output_path, metadata_path)
    except (FileNotFoundError, RuntimeError):
        return build_marcap_kospi200_proxy(source_path, output_path, metadata_path)
