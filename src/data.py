from pathlib import Path

import numpy as np
import pandas as pd

from .config import ALFRED_DIR, MARKET_SERIES_NAME, RAW_SERIES_CATALOG_FILE
from .raw_catalog import fred_config_from_catalog, load_raw_series_catalog
from .vintage import read_monthly_vintage_series


def read_fred_csv(path, name=None):

    path = Path(path)

    df = pd.read_csv(
        path,
        na_values=[".", "NA", "N/A", ""]
    )

    if "observation_date" not in df.columns:
        raise ValueError(
            f"{path.name}: observation_date 컬럼이 없어."
        )

    df["observation_date"] = pd.to_datetime(
        df["observation_date"]
    )

    value_cols = [
        c
        for c in df.columns
        if c != "observation_date"
    ]

    if len(value_cols) != 1:
        raise ValueError(
            f"{path.name}: 값 컬럼이 1개가 아니야: {value_cols}"
        )

    value_col = value_cols[0]

    values = (
        df[value_col]
        .astype(str)
        .str.replace(",", "", regex=False)
    )

    s = pd.to_numeric(
        values,
        errors="coerce"
    )

    s.index = df["observation_date"]

    s = s.sort_index()

    s.name = name or value_col

    return s


def infer_frequency(s):

    idx = s.dropna().index

    if len(idx) < 3:
        return "monthly"

    gaps = pd.Series(idx).diff().dt.days.dropna()

    median_gap = gaps.median()

    if median_gap <= 10:
        return "daily"

    return "monthly"


def to_monthly(
    s,
    freq,
    agg="last",
    availability_lag=0,
):

    s = s.sort_index()

    if freq not in {"daily", "weekly", "monthly", "quarterly"}:
        raise ValueError(f"Unsupported source frequency: {freq}")

    if freq in {"daily", "weekly"}:

        if agg == "mean":
            monthly = s.resample("ME").mean()

        elif agg == "last":
            monthly = s.resample("ME").last()

        elif agg == "sum":
            monthly = s.resample("ME").sum()

        else:
            raise ValueError(
                f"지원하지 않는 agg: {agg}"
            )

    else:

        periods = s.index.to_period("M")

        if agg == "mean":
            monthly = s.groupby(periods).mean()

        else:
            monthly = s.groupby(periods).last()

        monthly.index = monthly.index.to_timestamp(
            "M"
        )

    # --------------------------------------------------------
    # 발표 시점 처리
    #
    # 1월 CPI를 1월 말에 쓰면 안 됨.
    # lag=1 → 1월 관측치가 2월 말 정보로 들어감.
    # --------------------------------------------------------

    if availability_lag > 0:

        monthly.index = (
            monthly.index.to_period("M")
            + availability_lag
        ).to_timestamp("M")

    monthly = monthly.sort_index()
    if freq == "quarterly" and not monthly.empty:
        # The latest published quarter remains known through the next release.
        # Filling starts only after the configured availability lag.
        monthly = monthly.resample("ME").ffill()
    return monthly


def build_monthly_panel(fred_dir, catalog_path=RAW_SERIES_CATALOG_FILE):

    fred_dir = Path(fred_dir)

    all_series = []
    metadata = []

    catalog = load_raw_series_catalog(catalog_path)
    configured = fred_config_from_catalog(catalog)
    files = [
        fred_dir / filename
        for filename in configured
        if (fred_dir / filename).is_file()
    ]

    if not files:
        raise FileNotFoundError(
            f"CSV가 없어: {fred_dir}"
        )

    for path in files:

        config = configured.get(
            path.name,
            None
        )

        raw = read_fred_csv(path)

        if config is None:

            # 등록 안 된 FRED CSV도 자동으로 읽는다.
            # 다만 kind는 알 수 없으므로 index로 간주.
            freq = infer_frequency(raw)

            config = {
                "name": path.stem.lower(),
                "freq": freq,
                "agg": (
                    "mean"
                    if freq == "daily"
                    else "last"
                ),
                "kind": "index",
                "availability_lag": (
                    0
                    if freq == "daily"
                    else 1
                ),
                "group": "unknown",
            }

            print(
                f"[WARN] 자동 추론: {path.name} "
                f"→ {config}"
            )

        name = config["name"]

        raw.name = name
        vintage_path = ALFRED_DIR / f"{config['series_id']}_point_in_time.csv"
        if vintage_path.is_file():
            monthly = read_monthly_vintage_series(
                vintage_path, expected_series_id=config["series_id"]
            )
            alfred_vintage_used = True
            historical_revision_safe = True
            release_lag_applied = True
            availability_method = "observed_month_end_ALFRED_vintage"
        else:
            monthly = to_monthly(
                raw,
                freq=config["freq"],
                agg=config["agg"],
                availability_lag=config[
                    "availability_lag"
                ],
            )
            alfred_vintage_used = False
            historical_revision_safe = False
            release_lag_applied = True
            availability_method = "configured_month_lag_approximation"

        monthly.name = name

        all_series.append(monthly)

        metadata.append({
            "file": path.name,
            "source": "FRED",
            **config,
            "release_lag_applied": release_lag_applied,
            "alfred_vintage_used": alfred_vintage_used,
            "historical_revision_safe": historical_revision_safe,
            "availability_method": availability_method,
            "vintage_file": str(vintage_path) if alfred_vintage_used else None,
            "first_date": monthly.first_valid_index(),
            "last_date": monthly.last_valid_index(),
        })

    panel = pd.concat(
        all_series,
        axis=1
    ).sort_index()

    quarterly_names = [
        row["name"] for row in metadata if row["freq"] == "quarterly"
    ]
    if quarterly_names:
        panel[quarterly_names] = panel[quarterly_names].ffill()

    panel.index.name = "observation_date"

    metadata = pd.DataFrame(metadata)

    return panel, metadata


def _numeric_column(df, column):

    if column is None or column not in df.columns:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    return pd.to_numeric(
        df[column]
        .astype(str)
        .str.replace(",", "", regex=False),
        errors="coerce",
    )


def read_market_daily_csv(path):
    """Read daily OHLCV and repair the known shifted yfinance CSV.

    The current project file was exported from a yfinance DataFrame with
    a two-level header.  Its header has seven names but each data row has
    six fields, so pandas shifts the values as follows::

        Price=close, Close=high, High=low, Low=open, Open=volume

    This loader detects that shape from the empty Volume column and keeps
    the repair local; it never overwrites the source CSV.
    """

    path = Path(path)

    df = pd.read_csv(path)

    date_candidates = [
        "observation_date",
        "date",
        "Date",
        "DATE",
    ]

    date_col = next(
        (
            c
            for c in date_candidates
            if c in df.columns
        ),
        None
    )

    if date_col is None:
        raise ValueError(
            "시장 CSV에서 날짜 컬럼을 못 찾았어."
        )

    df[date_col] = pd.to_datetime(
        df[date_col]
    )

    empty_volume = (
        "Volume" in df.columns
        and _numeric_column(df, "Volume").notna().sum() == 0
    )
    shifted_yfinance = (
        "Price" in df.columns
        and all(
            column in df.columns
            for column in ["Close", "High", "Low", "Open"]
        )
        and empty_volume
    )

    if shifted_yfinance:
        print(
            "[WARN] KOSPI CSV의 밀린 yfinance 헤더 감지: "
            "Price→close, Close→high, High→low, "
            "Low→open, Open→volume으로 복구"
        )
        column_map = {
            "close": "Price",
            "high": "Close",
            "low": "High",
            "open": "Low",
            "volume": "Open",
        }
    else:
        aliases = {
            "close": [
                "close", "Close", "Adj Close", "adj_close",
                "Price", "price", "KOSPI",
            ],
            "high": ["high", "High"],
            "low": ["low", "Low"],
            "open": ["open", "Open"],
            "volume": ["volume", "Volume"],
        }
        column_map = {
            field: next(
                (
                    candidate
                    for candidate in candidates
                    if candidate in df.columns
                ),
                None,
            )
            for field, candidates in aliases.items()
        }

    if column_map["close"] is None:
        raise ValueError("시장 CSV에서 종가 컬럼을 못 찾았어.")

    daily = pd.DataFrame(
        {
            field: _numeric_column(
                df,
                column,
            ).to_numpy()
            for field, column in column_map.items()
        },
        index=pd.DatetimeIndex(
            df[date_col].to_numpy()
        ),
    )
    daily = daily.loc[daily["close"].notna()].sort_index()
    daily = daily.loc[~daily.index.duplicated(keep="last")]
    daily.index.name = "observation_date"

    if daily.empty:
        raise ValueError("시장 CSV에 유효한 종가가 없어.")

    ohlc_complete = daily[["open", "high", "low", "close"]].dropna()
    invalid_range = (
        (ohlc_complete["high"] < ohlc_complete[["open", "close"]].max(axis=1))
        | (ohlc_complete["low"] > ohlc_complete[["open", "close"]].min(axis=1))
        | (ohlc_complete["high"] < ohlc_complete["low"])
        | (ohlc_complete <= 0).any(axis=1)
    )
    if invalid_range.any():
        raise ValueError(
            f"Market OHLC range validation failed on {int(invalid_range.sum())} rows"
        )
    if (daily["volume"].dropna() < 0).any():
        raise ValueError("Market volume contains negative values")

    return daily


def read_investable_price_csv(path):
    """Read an audited total-return index or adjusted ETF price history.

    A close-only file is sufficient for portfolio returns.  Adjusted-close
    columns are preferred because distributions and splits must be reflected
    before this source can satisfy the strict operational gate.
    """
    path = Path(path)
    df = pd.read_csv(path)
    date_col = next(
        (name for name in ("observation_date", "date", "Date", "DATE") if name in df),
        None,
    )
    if date_col is None:
        raise ValueError("투자 가능 수익률 CSV에서 날짜 컬럼을 못 찾았어.")
    value_col = next(
        (
            name
            for name in (
                "adjusted_close",
                "adj_close",
                "Adj Close",
                "total_return_index",
                "close",
                "Close",
            )
            if name in df
        ),
        None,
    )
    if value_col is None:
        raise ValueError(
            "투자 가능 수익률 CSV에는 adjusted_close, total_return_index 또는 close가 필요해."
        )
    index = pd.to_datetime(df[date_col], errors="coerce")
    values = _numeric_column(df, value_col)
    series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(index), name="close")
    series = series.loc[series.index.notna() & series.notna()].sort_index()
    series = series.loc[~series.index.duplicated(keep="last")]
    if series.empty or (series <= 0).any():
        raise ValueError("투자 가능 수익률 CSV에 유효한 양수 가격지수가 없어.")
    series.attrs["value_column"] = value_col
    series.attrs["distribution_adjusted"] = value_col in {
        "adjusted_close", "adj_close", "Adj Close", "total_return_index"
    }
    return series


def read_market_csv(path):

    daily = read_market_daily_csv(path)
    price = daily["close"].resample("ME").last()
    price.name = MARKET_SERIES_NAME
    return price
