"""No-lookahead monthly ALFRED snapshots for market-rate series."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


FRED_GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def _month_ends(start, end):
    start = pd.Timestamp(start).to_period("M").to_timestamp("M")
    end = pd.Timestamp(end).to_period("M").to_timestamp("M")
    return pd.date_range(start, end, freq="ME")


def _download_csv(params, *, timeout=30, retries=8):
    url = FRED_GRAPH_URL + "?" + urlencode(params)
    error = None
    for attempt in range(retries):
        try:
            # FRED's CDN intermittently challenges Python's urllib client on
            # repeated requests.  The system curl client handles the CDN
            # cookie negotiation reliably; urllib remains the fallback.
            curl = shutil.which("curl")
            if curl:
                completed = subprocess.run(
                    [
                        curl,
                        "--fail",
                        "--location",
                        "--silent",
                        "--show-error",
                        "--max-time",
                        str(timeout),
                        url,
                    ],
                    check=False,
                    capture_output=True,
                    timeout=timeout + 5,
                )
                if completed.returncode == 0:
                    return completed.stdout.decode("utf-8-sig"), url
                raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))
            request = Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 EWS-vintage-audit/1.0"
                    ),
                    "Accept": "text/csv,*/*;q=0.8",
                },
            )
            with urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8-sig"), url
        except Exception as exc:  # network errors are retried and surfaced
            error = exc
            if attempt + 1 < retries:
                # The public graph endpoint rate-limits bursts with HTTP 403.
                # Back off rather than silently substituting a current value.
                time.sleep(min(2 ** attempt, 60) + random.random())
    raise RuntimeError(f"ALFRED download failed after {retries} attempts: {url}") from error


def fetch_month_end_vintage(series_id, month_end, *, agg="mean"):
    """Return the value observable by a completed month end."""
    month_end = pd.Timestamp(month_end).normalize()
    if agg == "latest_available":
        # Monthly and quarterly releases are dated for their reference period,
        # not their publication date.  A current-month-only query would miss a
        # January observation first released in February or March.  A bounded
        # lookback retrieves the latest reference-period value actually present
        # in the requested month-end vintage.
        observation_start = month_end - pd.DateOffset(months=24)
    else:
        observation_start = month_end.to_period("M").to_timestamp()
    text, url = _download_csv(
        {
            "id": series_id,
            "cosd": observation_start.date().isoformat(),
            "coed": month_end.date().isoformat(),
            "vintage_date": month_end.date().isoformat(),
        }
    )
    frame = pd.read_csv(io.StringIO(text), na_values=["."])
    if frame.shape[1] != 2 or "observation_date" not in frame.columns:
        raise ValueError(f"Unexpected ALFRED CSV schema for {series_id}: {frame.columns.tolist()}")
    value_column = next(column for column in frame.columns if column != "observation_date")
    frame["observation_date"] = pd.to_datetime(frame["observation_date"])
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    frame = frame.loc[
        frame["observation_date"].le(month_end),
        ["observation_date", value_column],
    ].dropna()
    if frame.empty:
        return {
            "observation_date": month_end,
            "value": np.nan,
            "vintage_date": month_end,
            "source_observations": 0,
            "source_first_date": pd.NaT,
            "source_last_date": pd.NaT,
            "source_url": url,
        }
    if agg == "mean":
        value = frame[value_column].mean()
    elif agg in {"last", "latest_available"}:
        value = frame[value_column].iloc[-1]
    elif agg == "sum":
        value = frame[value_column].sum()
    else:
        raise ValueError(f"Unsupported ALFRED aggregation: {agg}")
    return {
        "observation_date": month_end,
        "value": float(value),
        "vintage_date": month_end,
        "source_observations": len(frame),
        "source_first_date": frame["observation_date"].min(),
        "source_last_date": frame["observation_date"].max(),
        "source_url": url,
    }


def download_monthly_vintage_series(
    series_id,
    output_path,
    *,
    start,
    end,
    agg="mean",
    workers=1,
    request_interval_seconds=0.35,
):
    """Download each completed month using that month-end's ALFRED vintage.

    A partial file is checkpointed after every completed request.  This makes
    the deliberately rate-limited public-endpoint download safely resumable.
    The final audited file is only published after every requested month has
    been collected and validated.
    """
    output_path = Path(output_path)
    dates = _month_ends(start, end)
    metadata_path = output_path.with_suffix(".metadata.json")
    if output_path.is_file() and metadata_path.is_file():
        existing_values = read_monthly_vintage_series(
            output_path, expected_series_id=series_id
        )
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            existing_values.index.min() <= dates.min()
            and existing_values.index.max() >= dates.max()
            and existing_metadata.get("aggregation") == agg
        ):
            existing_frame = pd.read_csv(
                output_path,
                parse_dates=[
                    "observation_date",
                    "vintage_date",
                    "source_first_date",
                    "source_last_date",
                ],
            )
            return existing_frame, existing_metadata
    partial_path = output_path.with_suffix(".partial.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if partial_path.is_file():
        partial = pd.read_csv(
            partial_path,
            parse_dates=["observation_date", "vintage_date"],
        )
        if not partial.empty and not partial["series_id"].eq(series_id).all():
            raise ValueError(f"Partial ALFRED series id mismatch: {partial_path}")
        rows = partial.to_dict("records")
    else:
        rows = []
    completed = {
        pd.Timestamp(row["observation_date"]).normalize() for row in rows
    }
    missing_dates = [date for date in dates if date.normalize() not in completed]

    def throttled_fetch(date):
        if request_interval_seconds > 0:
            time.sleep(request_interval_seconds)
        return fetch_month_end_vintage(series_id, date, agg=agg)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(throttled_fetch, date): date
            for date in missing_dates
        }
        for future in as_completed(futures):
            row = future.result()
            row["series_id"] = series_id
            rows.append(row)
            pd.DataFrame(rows).sort_values("observation_date").to_csv(
                partial_path,
                index=False,
                quoting=csv.QUOTE_MINIMAL,
            )
    frame = pd.DataFrame(rows).sort_values("observation_date").reset_index(drop=True)
    if frame["observation_date"].duplicated().any():
        raise ValueError(f"Duplicate vintage month for {series_id}")
    if not frame["vintage_date"].eq(frame["observation_date"]).all():
        raise ValueError(f"Vintage date mismatch for {series_id}")
    if frame["value"].notna().sum() < 120:
        raise ValueError(f"Insufficient point-in-time history for {series_id}")
    if "series_id" not in frame:
        frame.insert(0, "series_id", series_id)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
    temporary.replace(output_path)
    partial_path.unlink(missing_ok=True)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    metadata = {
        "schema_version": 1,
        "series_id": series_id,
        "retrieval_method": "FRED graph vintage_date at each completed month end",
        "aggregation": agg,
        "start": frame["observation_date"].min().date().isoformat(),
        "end": frame["observation_date"].max().date().isoformat(),
        "months": len(frame),
        "non_missing_months": int(frame["value"].notna().sum()),
        "downloaded_at": datetime.now().astimezone().isoformat(),
        "sha256": digest,
        "point_in_time_safe": True,
        "execution_rule": "month-end information, next-month portfolio execution",
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return frame, metadata


def read_monthly_vintage_series(path, *, expected_series_id=None):
    path = Path(path)
    metadata_path = path.with_suffix(".metadata.json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"ALFRED vintage file or metadata missing: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != metadata.get("sha256"):
        raise RuntimeError(f"ALFRED vintage hash mismatch: {path}")
    if not metadata.get("point_in_time_safe"):
        raise RuntimeError(f"ALFRED vintage metadata is not approved as point-in-time: {path}")
    frame = pd.read_csv(
        path,
        parse_dates=["observation_date", "vintage_date", "source_first_date", "source_last_date"],
    )
    if expected_series_id and not frame["series_id"].eq(expected_series_id).all():
        raise ValueError(f"ALFRED series id mismatch: expected {expected_series_id}")
    if not frame["vintage_date"].eq(frame["observation_date"]).all():
        raise ValueError("ALFRED vintage dates must equal completed month ends")
    values = pd.Series(
        pd.to_numeric(frame["value"], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(frame["observation_date"]),
        name=expected_series_id or metadata["series_id"],
    ).sort_index()
    values.attrs.update(metadata)
    return values
