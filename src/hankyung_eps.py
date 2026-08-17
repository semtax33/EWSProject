"""Download and summarize Korean stock EPS estimates from Hankyung Consensus."""

from __future__ import annotations

import json
import time
from datetime import date
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


API_URL = "https://markets.hankyung.com/api/v2/consensus/search/report"
EPS_COLUMNS = [
    "report_id",
    "report_date",
    "stock_code",
    "stock_name",
    "brokerage",
    "analyst",
    "fiscal_period",
    "fiscal_year",
    "estimate_slot",
    "eps_estimate",
    "is_zero_eps",
]


class HankyungAPIError(RuntimeError):
    """Raised when the Hankyung Consensus API cannot be read."""


def _request_json(url, token, *, timeout, max_retries):
    request = Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "kr-eps-dispersion/1.0",
        },
        method="GET",
    )
    for attempt in range(max_retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt == max_retries:
                raise HankyungAPIError(
                    f"Hankyung API returned HTTP {exc.code}"
                ) from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == max_retries:
                raise HankyungAPIError(
                    f"Failed to read the Hankyung API: {exc}"
                ) from exc
        time.sleep(min(2**attempt, 8))
    raise AssertionError("retry loop exited unexpectedly")


def fetch_company_reports(
    token,
    from_date,
    to_date,
    *,
    page_size=1000,
    timeout=30,
    max_retries=3,
    request_delay=0.05,
):
    """Fetch all company reports in an inclusive publication-date range."""
    if not token or not str(token).strip():
        raise ValueError("A non-empty Hankyung API token is required")
    start = pd.Timestamp(from_date).date()
    end = pd.Timestamp(to_date).date()
    if start > end:
        raise ValueError("from_date must be on or before to_date")
    if not 1 <= int(page_size) <= 1000:
        raise ValueError("page_size must be between 1 and 1000")

    common_params = {
        "reportType": "CO",
        "fromDate": start.isoformat(),
        "toDate": end.isoformat(),
        "gradeCode": "ALL",
        "changePrices": "ALL",
        "searchType": "ALL",
        "reportRange": int(page_size),
    }
    rows = []
    page = 1
    while True:
        params = {"page": page, **common_params}
        payload = _request_json(
            f"{API_URL}?{urlencode(params)}",
            str(token).strip(),
            timeout=timeout,
            max_retries=max_retries,
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise HankyungAPIError("Unexpected Hankyung API response schema")
        rows.extend(payload["data"])

        try:
            last_page = int(payload.get("last_page", page))
        except (TypeError, ValueError) as exc:
            raise HankyungAPIError("Invalid last_page in API response") from exc
        if page >= last_page:
            break
        page += 1
        if request_delay > 0:
            time.sleep(request_delay)
    return rows


def _text(value):
    if value is None:
        return ""
    return str(value).strip()


def _fiscal_period(value):
    """Return YYYY or YYYYMM from the API's mixed period representation."""
    digits = "".join(character for character in _text(value) if character.isdigit())
    if len(digits) >= 6:
        period = digits[:6]
        month = int(period[4:6])
        return period if 1 <= month <= 12 else ""
    return digits if len(digits) == 4 else ""


def _number(value):
    text = _text(value).replace(",", "").replace("원", "").replace(" ", "")
    if not text or text.lower() in {"n/a", "na", "null", "-", "--"}:
        return np.nan
    if text.startswith("(") and text.endswith(")"):
        text = f"-{text[1:-1]}"
    try:
        number = float(text)
    except ValueError:
        return np.nan
    return number if np.isfinite(number) else np.nan


def normalize_eps_estimates(reports):
    """Convert both legacy and current API fields to one estimate per row."""
    normalized = []
    for report in reports:
        raw_stock_code = _text(report.get("BUSINESS_CODE"))
        stock_code = (
            raw_stock_code.zfill(6)
            if raw_stock_code.isdigit() and len(raw_stock_code) <= 6
            else ""
        )
        base = {
            "report_id": _text(report.get("REPORT_IDX")),
            "report_date": report.get("REPORT_DATE"),
            "stock_code": stock_code,
            "stock_name": _text(report.get("BUSINESS_NAME")),
            "brokerage": _text(report.get("OFFICE_NAME")),
            "analyst": _text(report.get("REPORT_WRITER")),
        }
        estimates_by_period = {}

        current_period = _fiscal_period(report.get("STOCK_SETTLEMENT_DAY"))
        current_eps = _number(report.get("STOCK_PRE_EPS"))
        if current_period and np.isfinite(current_eps):
            estimates_by_period[current_period] = ("primary", current_eps)

        for slot in range(1, 4):
            period = _fiscal_period(report.get(f"STOCK_SETTLEMENT_DAY{slot}"))
            eps = _number(report.get(f"STOCK_EPS{slot}"))
            if period and np.isfinite(eps) and period not in estimates_by_period:
                estimates_by_period[period] = (f"legacy_{slot}", eps)

        for period, (slot, eps) in estimates_by_period.items():
            normalized.append(
                {
                    **base,
                    "fiscal_period": period,
                    "fiscal_year": int(period[:4]),
                    "estimate_slot": slot,
                    "eps_estimate": eps,
                    "is_zero_eps": eps == 0.0,
                }
            )

    frame = pd.DataFrame(normalized, columns=EPS_COLUMNS)
    if frame.empty:
        return frame
    frame["report_date"] = pd.to_datetime(frame["report_date"], errors="coerce")
    frame["eps_estimate"] = pd.to_numeric(frame["eps_estimate"], errors="coerce")
    frame = frame.loc[
        frame["report_date"].notna()
        & frame["stock_code"].str.fullmatch(r"\d{6}", na=False)
        & frame["eps_estimate"].notna()
    ].copy()
    return frame.sort_values(
        ["report_date", "report_id", "stock_code", "fiscal_period"],
        kind="stable",
    ).reset_index(drop=True)


def _latest_nonempty(values):
    nonempty = values.loc[values.astype("string").str.strip().ne("")]
    return nonempty.iloc[-1] if not nonempty.empty else ""


def calculate_eps_dispersion(
    estimates,
    *,
    dedupe_by="brokerage",
    min_estimates=2,
    ddof=1,
    exclude_zero=True,
):
    """Calculate cross-sectional EPS dispersion by stock and fiscal period.

    By default only the latest report from each brokerage in the input window
    contributes to a stock-period consensus. ``ddof=1`` gives sample standard
    deviation; pass ``ddof=0`` for population standard deviation. Exact-zero
    EPS values are excluded by default because the current API commonly uses
    them as unavailable-value placeholders. They remain in the normalized
    estimate output for auditing.
    """
    required = set(EPS_COLUMNS)
    missing = required.difference(estimates.columns)
    if missing:
        raise ValueError(f"Missing estimate columns: {sorted(missing)}")
    if dedupe_by not in {"brokerage", "brokerage_analyst", "none"}:
        raise ValueError("dedupe_by must be brokerage, brokerage_analyst, or none")
    if min_estimates < 1:
        raise ValueError("min_estimates must be at least 1")
    if ddof not in {0, 1}:
        raise ValueError("ddof must be 0 or 1")

    frame = estimates.copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "stock_code",
                "stock_name",
                "fiscal_period",
                "fiscal_year",
                "as_of_date",
                "first_report_date",
                "n_estimates",
                "n_brokerages",
                "eps_mean",
                "eps_median",
                "eps_stddev",
                "eps_min",
                "eps_max",
                "eps_range",
                "relative_stddev",
            ]
        )

    frame["report_date"] = pd.to_datetime(frame["report_date"], errors="coerce")
    frame["eps_estimate"] = pd.to_numeric(frame["eps_estimate"], errors="coerce")
    frame = frame.dropna(subset=["report_date", "eps_estimate"])
    if exclude_zero:
        frame = frame.loc[frame["eps_estimate"].ne(0.0)].copy()
    frame["_report_order"] = pd.to_numeric(frame["report_id"], errors="coerce")
    frame = frame.sort_values(
        ["report_date", "_report_order", "report_id"], kind="stable"
    )

    identity = ["stock_code", "fiscal_period"]
    if dedupe_by == "brokerage":
        frame = frame.drop_duplicates(identity + ["brokerage"], keep="last")
    elif dedupe_by == "brokerage_analyst":
        frame = frame.drop_duplicates(
            identity + ["brokerage", "analyst"], keep="last"
        )

    groups = frame.groupby(identity, sort=True, dropna=False)
    summary = groups.agg(
        stock_name=("stock_name", _latest_nonempty),
        fiscal_year=("fiscal_year", "last"),
        as_of_date=("report_date", "max"),
        first_report_date=("report_date", "min"),
        n_estimates=("eps_estimate", "size"),
        n_brokerages=("brokerage", "nunique"),
        eps_mean=("eps_estimate", "mean"),
        eps_median=("eps_estimate", "median"),
        eps_min=("eps_estimate", "min"),
        eps_max=("eps_estimate", "max"),
    ).reset_index()
    stddev = groups["eps_estimate"].std(ddof=ddof).rename("eps_stddev")
    summary = summary.merge(stddev.reset_index(), on=identity, how="left")
    summary["eps_range"] = summary["eps_max"] - summary["eps_min"]
    nonzero_mean = summary["eps_mean"].abs().replace(0.0, np.nan)
    summary["relative_stddev"] = summary["eps_stddev"] / nonzero_mean
    summary = summary.loc[summary["n_estimates"] >= min_estimates].copy()
    ordered = [
        "stock_code",
        "stock_name",
        "fiscal_period",
        "fiscal_year",
        "as_of_date",
        "first_report_date",
        "n_estimates",
        "n_brokerages",
        "eps_mean",
        "eps_median",
        "eps_stddev",
        "eps_min",
        "eps_max",
        "eps_range",
        "relative_stddev",
    ]
    return summary[ordered].sort_values(
        ["fiscal_period", "stock_code"], kind="stable"
    ).reset_index(drop=True)


def download_eps_dispersion(
    token,
    from_date,
    to_date: date | str,
    *,
    dedupe_by="brokerage",
    min_estimates=2,
    ddof=1,
    exclude_zero=True,
    **fetch_kwargs,
):
    """Fetch reports and return ``(normalized_estimates, dispersion)``."""
    reports = fetch_company_reports(
        token, from_date, to_date, **fetch_kwargs
    )
    estimates = normalize_eps_estimates(reports)
    dispersion = calculate_eps_dispersion(
        estimates,
        dedupe_by=dedupe_by,
        min_estimates=min_estimates,
        ddof=ddof,
        exclude_zero=exclude_zero,
    )
    return estimates, dispersion
