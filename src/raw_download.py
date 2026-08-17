"""Download and validate catalogued FRED CSV files."""

from __future__ import annotations

import io
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


FRED_GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def _validate_payload(payload, series_id):
    frame = pd.read_csv(io.BytesIO(payload), na_values=[".", "NA", "N/A", ""])
    if "observation_date" not in frame.columns or len(frame.columns) != 2:
        raise ValueError(f"{series_id}: unexpected FRED CSV schema")
    dates = pd.to_datetime(frame["observation_date"], errors="coerce")
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    if dates.notna().sum() < 2 or values.notna().sum() < 2:
        raise ValueError(f"{series_id}: fewer than two valid observations")
    if not dates.dropna().is_monotonic_increasing:
        raise ValueError(f"{series_id}: dates are not sorted")
    return frame, dates, values


def download_fred_series(
    series_id,
    destination,
    *,
    timeout=45,
    retries=3,
    opener=urllib.request.urlopen,
):
    """Download one series atomically; return validated coverage metadata."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = FRED_GRAPH_URL.format(series_id=series_id)
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "EWS-Research/1.0 (+local reproducible study)"},
            )
            with opener(request, timeout=timeout) as response:
                payload = response.read()
            frame, dates, values = _validate_payload(payload, series_id)
            temp_path = destination.with_suffix(destination.suffix + ".part")
            temp_path.write_bytes(payload)
            temp_path.replace(destination)
            valid_dates = dates.loc[values.notna()]
            return {
                "series_id": series_id,
                "file": destination.name,
                "status": "downloaded",
                "observations": int(values.notna().sum()),
                "first_date": valid_dates.min(),
                "last_date": valid_dates.max(),
                "bytes": len(payload),
                "error": None,
            }
        except Exception as error:  # surfaced in the machine-readable report
            last_error = error
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 4))
    return {
        "series_id": series_id,
        "file": destination.name,
        "status": "failed",
        "observations": 0,
        "first_date": None,
        "last_date": None,
        "bytes": 0,
        "error": f"{type(last_error).__name__}: {last_error}",
    }


def download_catalog(catalog, output_dir, refresh=False, workers=8, **download_kwargs):
    """Download enabled FRED rows and preserve an explicit per-series report."""
    output_dir = Path(output_dir)
    fred = catalog.loc[catalog["enabled"] & catalog["source"].eq("FRED")]

    def fetch(row):
        destination = output_dir / row.file
        if destination.exists() and not refresh:
            try:
                payload = destination.read_bytes()
                _, dates, values = _validate_payload(payload, row.series_id)
                valid_dates = dates.loc[values.notna()]
                result = {
                    "series_id": row.series_id,
                    "file": row.file,
                    "status": "existing_valid",
                    "observations": int(values.notna().sum()),
                    "first_date": valid_dates.min(),
                    "last_date": valid_dates.max(),
                    "bytes": destination.stat().st_size,
                    "error": None,
                }
            except Exception:
                result = download_fred_series(
                    row.series_id, destination, **download_kwargs
                )
        else:
            result = download_fred_series(
                row.series_id, destination, **download_kwargs
            )
        return {"name": row.name, "group": row.group, **result}

    catalog_rows = list(fred.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        rows = list(executor.map(fetch, catalog_rows))
    for row in rows:
        print(f"[{row['status']}] {row['series_id']} -> {row['file']}")
    return pd.DataFrame(rows)
