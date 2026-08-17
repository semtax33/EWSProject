"""Declarative 70-sensor raw-data universe and coverage validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


EXPECTED_GROUP_COUNTS = {
    "cycle": 10,
    "liquidity": 10,
    "credit": 8,
    "earnings_valuation": 12,
    "kospi_internal": 12,
    "global_risk": 10,
    "sentiment_uncertainty": 8,
}

REQUIRED_COLUMNS = {
    "group",
    "source",
    "series_id",
    "file",
    "name",
    "freq",
    "agg",
    "kind",
    "availability_lag",
    "sensor_family",
    "factory_mode",
    "required_for_target",
    "enabled",
}


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def load_raw_series_catalog(path, validate_counts=True):
    """Read and validate the predeclared raw-series catalog.

    The catalog is intentionally outcome-free.  Names, groups and publication
    lags must be fixed before any target-based screening is run.
    """
    path = Path(path)
    catalog = pd.read_csv(path, keep_default_na=False)
    missing = REQUIRED_COLUMNS.difference(catalog.columns)
    if missing:
        raise ValueError(f"Raw-series catalog columns missing: {sorted(missing)}")

    catalog = catalog.copy()
    for column in ("required_for_target", "enabled"):
        catalog[column] = catalog[column].map(_parse_bool)
    catalog["availability_lag"] = pd.to_numeric(
        catalog["availability_lag"], errors="raise"
    ).astype(int)

    enabled = catalog.loc[catalog["enabled"]].copy()
    if enabled["name"].duplicated().any():
        duplicates = enabled.loc[enabled["name"].duplicated(), "name"].tolist()
        raise ValueError(f"Duplicate raw-series names: {duplicates}")
    if (enabled["availability_lag"] < 0).any():
        raise ValueError("availability_lag must be non-negative")
    if not enabled["factory_mode"].isin({"full", "curated"}).all():
        raise ValueError("factory_mode must be full or curated")

    fred = enabled.loc[enabled["source"].eq("FRED")]
    if fred["series_id"].eq("").any() or fred["file"].eq("").any():
        raise ValueError("Every enabled FRED row needs series_id and file")
    if fred["series_id"].duplicated().any():
        raise ValueError("Enabled FRED series_id values must be unique")

    if validate_counts:
        counts = enabled.groupby("group").size().to_dict()
        if counts != EXPECTED_GROUP_COUNTS:
            raise ValueError(
                "Raw-series group counts do not match the 70-sensor design: "
                f"expected={EXPECTED_GROUP_COUNTS}, actual={counts}"
            )
    return catalog


def fred_config_from_catalog(catalog):
    """Return the filename-keyed configuration consumed by the panel loader."""
    fred = catalog.loc[catalog["enabled"] & catalog["source"].eq("FRED")]
    fields = [
        "name",
        "freq",
        "agg",
        "kind",
        "availability_lag",
        "group",
        "series_id",
        "sensor_family",
        "factory_mode",
        "required_for_target",
    ]
    return {
        row["file"]: {field: row[field] for field in fields}
        for _, row in fred.iterrows()
    }


def build_raw_series_coverage(
    catalog,
    fred_dir,
    available_derived_names=(),
    *,
    research_start="2000-03-31",
    research_end="2014-03-31",
    minimum_observations=120,
):
    """Report availability without pretending that planned data are installed."""
    fred_dir = Path(fred_dir)
    derived_frame = (
        available_derived_names
        if isinstance(available_derived_names, pd.DataFrame)
        else None
    )
    available_derived_names = set(
        derived_frame.columns if derived_frame is not None else available_derived_names
    )
    rows = []
    for row in catalog.loc[catalog["enabled"]].itertuples(index=False):
        if row.source == "FRED":
            path = fred_dir / row.file
            available = path.is_file() and path.stat().st_size > 0
            local_path = str(path)
            if available:
                try:
                    frame = pd.read_csv(
                        path, na_values=[".", "NA", "N/A", ""]
                    )
                    dates = pd.to_datetime(frame["observation_date"], errors="coerce")
                    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
                    valid_dates = dates.loc[values.notna() & dates.notna()]
                except Exception:
                    available = False
                    valid_dates = pd.Series(dtype="datetime64[ns]")
            else:
                valid_dates = pd.Series(dtype="datetime64[ns]")
        else:
            available = row.name in available_derived_names
            local_path = None
            if available and derived_frame is not None:
                valid_dates = pd.Series(
                    derived_frame[row.name].dropna().index,
                    dtype="datetime64[ns]",
                )
            else:
                valid_dates = pd.Series(dtype="datetime64[ns]")
        observations = int(len(valid_dates))
        first_date = valid_dates.min() if observations else pd.NaT
        last_date = valid_dates.max() if observations else pd.NaT
        research_eligible = bool(
            available
            and observations >= minimum_observations
            and first_date <= pd.Timestamp(research_start)
            and last_date >= pd.Timestamp(research_end)
        )
        rows.append(
            {
                **row._asdict(),
                "available": bool(available),
                "research_eligible": research_eligible,
                "observations": observations,
                "first_date": first_date,
                "last_date": last_date,
                "status": (
                    "research_eligible"
                    if research_eligible
                    else "insufficient_history"
                    if available
                    else "missing"
                ),
                "local_path": local_path,
            }
        )
    return pd.DataFrame(rows)


def assert_expanded_universe(coverage, minimum_available=50):
    """Fail early if a purported expanded run still uses the legacy universe."""
    available = coverage.loc[coverage["research_eligible"]]
    required_missing = coverage.loc[
        coverage["required_for_target"] & ~coverage["research_eligible"], "name"
    ].tolist()
    if required_missing:
        raise FileNotFoundError(
            "Required raw series are missing: " + ", ".join(required_missing)
        )
    if len(available) < minimum_available:
        raise RuntimeError(
            f"Expanded universe requires at least {minimum_available} of 70 raw "
            f"series, but only {len(available)} have sufficient research history. Run "
            "`python download_raw_series.py` first, or use "
            "`--allow-partial-raw-universe` only for a smoke test."
        )
