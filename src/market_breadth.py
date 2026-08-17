"""Reproducible market-breadth factors from the Korean stock panel.

The source is a large point-in-date stock-universe panel.  Processing is
chunked so the 2GB CSV does not need to fit in memory.  Indicators that need
KOSPI200 membership are deliberately not claimed here: these are exact for
the supplied Korean listed-stock universe, not substitutes for a historical
KOSPI200 constituent panel.
"""

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
    "Close",
    "ChangesRatio",
    "Volume",
    "Amount",
    "Marcap",
    "Stocks",
    "Market",
    "observation_date",
]


def _average_pairwise_correlation(month_returns, min_days=10):
    """Average off-diagonal correlation without materializing an NxN matrix."""
    matrix = month_returns.pivot(
        index="observation_date", columns="Code", values="return_1d"
    )
    matrix = matrix.dropna(axis=1, how="any")
    if matrix.shape[0] < min_days or matrix.shape[1] < 2:
        return np.nan, matrix.shape[1], matrix.shape[0]
    std = matrix.std(axis=0, ddof=1)
    matrix = matrix.loc[:, std > 0]
    n_assets = matrix.shape[1]
    if n_assets < 2:
        return np.nan, n_assets, matrix.shape[0]
    standardized = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0, ddof=1)
    summed_by_day = standardized.sum(axis=1).to_numpy()
    total_correlation_sum = (
        np.square(summed_by_day).sum() / (matrix.shape[0] - 1)
    )
    average_off_diagonal = (
        total_correlation_sum - n_assets
    ) / (n_assets * (n_assets - 1))
    return float(average_off_diagonal), n_assets, matrix.shape[0]


def _sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def build_market_breadth(
    source_path,
    output_path,
    metadata_path,
    *,
    chunksize=500_000,
):
    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    metadata_path = Path(metadata_path).resolve()
    source_stat = source_path.stat()

    monthly_parts = []
    daily_parts = []
    pending_daily_returns = None
    pairwise_rows = []
    source_rows = 0
    first_date = None
    last_date = None

    reader = pd.read_csv(
        source_path,
        usecols=SOURCE_COLUMNS,
        dtype={
            "Code": "string",
            "Close": "float64",
            "ChangesRatio": "float64",
            "Volume": "float64",
            "Amount": "float64",
            "Marcap": "float64",
            "Stocks": "float64",
            "Market": "string",
        },
        parse_dates=["observation_date"],
        chunksize=chunksize,
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        chunk = chunk.dropna(subset=["Code", "observation_date"]).copy()
        chunk = chunk.loc[chunk["Market"].eq("KOSPI")].copy()
        chunk = chunk.sort_values(["observation_date", "Code"], kind="stable")
        source_rows += len(chunk)
        chunk_first = chunk["observation_date"].min()
        chunk_last = chunk["observation_date"].max()
        first_date = chunk_first if first_date is None else min(first_date, chunk_first)
        last_date = chunk_last if last_date is None else max(last_date, chunk_last)

        valid_return = (chunk["ChangesRatio"] / 100.0).replace(
            [np.inf, -np.inf], np.nan
        )

        return_rows = pd.DataFrame(
            {
                "observation_date": chunk["observation_date"],
                "Code": chunk["Code"],
                "return_1d": valid_return,
            }
        )
        pending_daily_returns = (
            return_rows.reset_index(drop=True)
            if pending_daily_returns is None
            else pd.concat([pending_daily_returns, return_rows], ignore_index=True)
        )
        return_month = pending_daily_returns["observation_date"].dt.to_period("M")
        final_month = return_month.max()
        completed = pending_daily_returns.loc[return_month < final_month]
        if not completed.empty:
            for completed_month, month_returns in completed.groupby(
                completed["observation_date"].dt.to_period("M"), sort=True
            ):
                correlation, assets, days = _average_pairwise_correlation(
                    month_returns.dropna(subset=["return_1d"])
                )
                pairwise_rows.append(
                    {
                        "month": completed_month,
                        "pairwise_correlation": correlation,
                        "pairwise_assets": assets,
                        "pairwise_days": days,
                    }
                )
        pending_daily_returns = pending_daily_returns.loc[
            return_month == final_month
        ].copy()

        direction = pd.DataFrame(
            {
                "observation_date": chunk["observation_date"],
                "advances": (valid_return > 0).astype("int32"),
                "declines": (valid_return < 0).astype("int32"),
                "unchanged": (valid_return == 0).astype("int32"),
                "valid_returns": valid_return.notna().astype("int32"),
                "amount": chunk["Amount"].fillna(0.0),
                "marcap": chunk["Marcap"].fillna(0.0),
                "volume": chunk["Volume"].fillna(0.0),
                "stocks": chunk["Stocks"].fillna(0.0),
            }
        )
        daily_parts.append(
            direction.groupby("observation_date", sort=True).sum(numeric_only=True)
        )

        month = chunk["observation_date"].dt.to_period("M")
        monthly_chunk = (
            chunk.assign(month=month, return_gross=1.0 + valid_return)
            .groupby(["month", "Code"], sort=False, observed=True)
            .agg(
                close=("Close", "last"),
                volume=("Volume", "sum"),
                amount=("Amount", "sum"),
                return_gross=("return_gross", "prod"),
            )
        )
        monthly_parts.append(monthly_chunk)

        print(f"market breadth chunk {chunk_number:,}: {source_rows:,} rows")

    if pending_daily_returns is not None and not pending_daily_returns.empty:
        for completed_month, month_returns in pending_daily_returns.groupby(
            pending_daily_returns["observation_date"].dt.to_period("M"), sort=True
        ):
            correlation, assets, days = _average_pairwise_correlation(
                month_returns.dropna(subset=["return_1d"])
            )
            pairwise_rows.append(
                {
                    "month": completed_month,
                    "pairwise_correlation": correlation,
                    "pairwise_assets": assets,
                    "pairwise_days": days,
                }
            )

    daily = pd.concat(daily_parts).groupby(level=0).sum().sort_index()
    denominator = (daily["advances"] + daily["declines"]).replace(0, np.nan)
    daily["advance_decline_ratio"] = daily["advances"] / daily["declines"].replace(0, np.nan)
    daily["normalized_net_advances"] = (
        100.0 * (daily["advances"] - daily["declines"]) / denominator
    )
    daily["mcclellan_oscillator"] = (
        daily["normalized_net_advances"].ewm(span=19, adjust=False).mean()
        - daily["normalized_net_advances"].ewm(span=39, adjust=False).mean()
    )

    monthly_panel = pd.concat(monthly_parts)
    monthly_panel = (
        monthly_panel.groupby(level=[0, 1], sort=True, observed=True)
        .agg(
            close=("close", "last"),
            volume=("volume", "sum"),
            amount=("amount", "sum"),
            return_gross=("return_gross", "prod"),
        )
        .reset_index()
        .sort_values(["Code", "month"])
    )
    monthly_panel["return_1m"] = monthly_panel["return_gross"] - 1.0

    valid_monthly_returns = monthly_panel.dropna(subset=["return_1m"]).copy()
    return_groups = valid_monthly_returns.groupby("month", observed=True)["return_1m"]
    lower = return_groups.transform(lambda values: values.quantile(0.01))
    upper = return_groups.transform(lambda values: values.quantile(0.99))
    valid_monthly_returns["robust_return_1m"] = valid_monthly_returns[
        "return_1m"
    ].clip(lower=lower, upper=upper)
    robust_groups = valid_monthly_returns.groupby("month", observed=True)[
        "robust_return_1m"
    ]
    factors = pd.DataFrame(
        {
            "korea_stock_universe_return_skew_1m": robust_groups.skew(),
            "korea_stock_universe_return_dispersion_1m": robust_groups.std(),
            "korea_stock_universe_return_skew_1m_raw": return_groups.skew(),
            "korea_stock_universe_return_dispersion_1m_raw": return_groups.std(),
            "korea_stock_universe_return_up_ratio_1m": return_groups.apply(
                lambda values: float((values > 0).mean())
            ),
            "korea_stock_universe_return_observations": return_groups.size(),
        }
    )

    total_volume = monthly_panel.groupby("month", observed=True)["volume"].sum(min_count=1)
    factors["korea_stock_universe_trading_volume_ratio_12m"] = total_volume / (
        total_volume.shift(1).rolling(12, min_periods=6).mean()
    )
    daily["trading_value_to_market_cap"] = (
        daily["amount"] / daily["marcap"].replace(0, np.nan)
    )
    daily["share_turnover_ratio"] = (
        daily["volume"] / daily["stocks"].replace(0, np.nan)
    )
    daily_monthly = daily.resample("ME").last()
    daily_monthly.index = daily_monthly.index.to_period("M")
    factors["korea_stock_universe_advance_decline_ratio"] = daily_monthly[
        "advance_decline_ratio"
    ]
    factors["korea_stock_universe_mcclellan_oscillator"] = daily_monthly[
        "mcclellan_oscillator"
    ]
    factors["korea_stock_universe_trading_value_to_market_cap"] = (
        daily["trading_value_to_market_cap"].resample("ME").sum().set_axis(
            daily_monthly.index
        )
    )
    factors["korea_stock_universe_share_turnover_ratio"] = (
        daily["share_turnover_ratio"].resample("ME").sum().set_axis(
            daily_monthly.index
        )
    )
    pairwise = pd.DataFrame(pairwise_rows).drop_duplicates("month", keep="last")
    pairwise = pairwise.set_index("month").sort_index()
    factors["korea_stock_universe_pairwise_correlation_1m"] = pairwise[
        "pairwise_correlation"
    ]
    factors["korea_stock_universe_pairwise_assets"] = pairwise["pairwise_assets"]

    factors.index = factors.index.to_timestamp("M")
    factors.index.name = "observation_date"
    factors = factors.replace([np.inf, -np.inf], np.nan).sort_index()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    factors.to_parquet(output_path)

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
        "source_first_date": str(first_date.date()),
        "source_last_date": str(last_date.date()),
        "universe": "KOSPI stocks in the supplied Korean listed-stock panel",
        "kospi200_membership_available": False,
        "output": str(output_path),
        "output_sha256": output_hash,
        "output_rows": len(factors),
        "output_columns": list(factors.columns),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return factors, metadata


def load_market_breadth(source_path, output_path, metadata_path):
    """Load the cache only if it still corresponds to the source file."""
    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    metadata_path = Path(metadata_path).resolve()
    if not output_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "Market-breadth cache missing; run build_market_breadth.py first"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_stat = source_path.stat()
    if (
        metadata.get("source_bytes") != source_stat.st_size
        or metadata.get("source_mtime_ns") != source_stat.st_mtime_ns
    ):
        raise RuntimeError(
            "Korean stock panel changed after the breadth cache was built; rebuild it"
        )
    factors = pd.read_parquet(output_path).sort_index()
    if list(factors.columns) != metadata.get("output_columns"):
        raise RuntimeError("Market-breadth cache schema differs from its metadata")
    return factors, metadata


def market_breadth_metadata(factors):
    descriptions = {
        "korea_stock_universe_return_skew_1m": "월말 종목별 1개월 수익률의 횡단면 왜도",
        "korea_stock_universe_return_dispersion_1m": "월말 종목별 1개월 수익률의 횡단면 표준편차; 월별 1/99% winsorization",
        "korea_stock_universe_return_skew_1m_raw": "품질진단용 비 winsorized 횡단면 왜도",
        "korea_stock_universe_return_dispersion_1m_raw": "품질진단용 비 winsorized 횡단면 표준편차",
        "korea_stock_universe_return_up_ratio_1m": "월말 종목별 1개월 수익률 양수 비율",
        "korea_stock_universe_return_observations": "횡단면 수익률 계산에 사용된 종목 수",
        "korea_stock_universe_trading_volume_ratio_12m": "월 거래량 합계 / 직전 12개월 평균",
        "korea_stock_universe_advance_decline_ratio": "월말 일간 상승종목 수 / 하락종목 수",
        "korea_stock_universe_mcclellan_oscillator": "정규화 순상승종목의 19일 EMA - 39일 EMA",
        "korea_stock_universe_trading_value_to_market_cap": "월중 일별 거래대금/시가총액 비율의 합",
        "korea_stock_universe_share_turnover_ratio": "월중 일별 거래량/상장주식수 비율의 합",
        "korea_stock_universe_pairwise_correlation_1m": "월중 완전관측 종목 일수익률의 평균 pairwise correlation",
        "korea_stock_universe_pairwise_assets": "pairwise correlation에 사용된 종목 수",
    }
    rows = []
    for feature in factors.columns:
        rows.append(
            {
                "feature": feature,
                "base": feature,
                "source": "Korean listed-stock daily panel",
                "market": "Korean listed-stock universe",
                "ticker": None,
                "group": "market_breadth",
                "exactness": "direct_universe",
                "model_eligible": feature not in {
                    "korea_stock_universe_return_observations",
                    "korea_stock_universe_pairwise_assets",
                    "korea_stock_universe_return_skew_1m_raw",
                    "korea_stock_universe_return_dispersion_1m_raw",
                },
                "availability_lag": 0,
                "point_in_time_rule": "same-month observations, next-month execution",
                "description": descriptions[feature],
                "first_date": factors[feature].first_valid_index(),
                "last_date": factors[feature].last_valid_index(),
                "observations": int(factors[feature].notna().sum()),
            }
        )
    return pd.DataFrame(rows)
