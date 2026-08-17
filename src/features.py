import numpy as np
import pandas as pd
import re

from .config import (
    CHANGE_HORIZONS,
    MA_WINDOWS,
    Z_WINDOWS,
    EWM_SPANS,
    VOL_WINDOWS,
    SLOPE_WINDOWS,
)


RATE_LIKE = {
    "rate",
    "spread",
}


COMPACT_TRANSFORM_PATTERNS = (
    r"level",
    r"chg_(1|3|6|12|24)m",
    r"dist_ma_(3|6|12|24)m",
    r"z_(12|36|60)m",
    r"dist_ewma_(3|6|12)m",
    r"vol_(6|12|24)m",
    r"slope_(6|12|24)m",
)


def transformation_family(feature):
    """Return a stable family label used to control near-duplicate search."""
    if "__" not in feature:
        return "market_curated"
    transform = feature.split("__", 1)[1]
    if transform == "level":
        return "level"
    if transform.startswith("chg_"):
        return "change"
    if transform.startswith("dist_ma_"):
        return "ma_distance"
    if transform.startswith("ma_") and "_chg_" in transform:
        return "ma_change"
    if transform.startswith("ma_"):
        return "ma_level"
    if transform.startswith("z_"):
        return "zscore"
    if transform.startswith("dist_ewma_"):
        return "ewma_distance"
    if transform.startswith("ewma_") and "_chg_" in transform:
        return "ewma_change"
    if transform.startswith("ewma_"):
        return "ewma_level"
    if transform.startswith("vol_"):
        return "volatility"
    if transform.startswith("slope_"):
        return "trend"
    return "other"


def compact_candidate_columns(columns):
    """Pre-declared, outcome-independent subset for nested feature search.

    The full Factor Factory remains available for diagnostics.  Repeated
    outer-fold selection uses this compact family registry to reduce the
    multiple-testing burden before any outcomes in a fold are inspected.
    """
    selected = []
    for feature in columns:
        if "__" not in feature:
            selected.append(feature)
            continue
        transform = feature.split("__", 1)[1]
        if any(re.fullmatch(pattern, transform) for pattern in COMPACT_TRANSFORM_PATTERNS):
            selected.append(feature)
    return selected


def change(s, horizon, kind):

    if kind in RATE_LIKE:

        # 4% → 5%
        # +25%가 아니라 +1%p로 해석
        return s.diff(horizon)

    previous = s.shift(horizon)

    out = (
        s / previous
    ) - 1

    return out


def distance_from_average(
    s,
    avg,
    kind,
):

    if kind in RATE_LIKE:

        return s - avg

    return (
        s / avg
    ) - 1


def rolling_zscore(
    s,
    window,
):

    mean = s.rolling(
        window,
        min_periods=window,
    ).mean()

    std = s.rolling(
        window,
        min_periods=window,
    ).std()

    return (
        s - mean
    ) / std


def rolling_slope(
    s,
    window,
):

    def calc(arr):

        if np.isnan(arr).any():
            return np.nan

        x = np.arange(
            len(arr),
            dtype=float
        )

        return np.polyfit(
            x,
            arr,
            1
        )[0]

    return s.rolling(
        window,
        min_periods=window,
    ).apply(
        calc,
        raw=True,
    )


def factor_factory(
    s,
    name,
    kind,
):

    factors = {}

    # ========================================================
    # 1. LEVEL
    # ========================================================

    factors[
        f"{name}__level"
    ] = s


    # ========================================================
    # 2. 원자료 변화
    # ========================================================

    for h in CHANGE_HORIZONS:

        factors[
            f"{name}__chg_{h}m"
        ] = change(
            s,
            h,
            kind,
        )


    # ========================================================
    # 3. 이동평균 + 이동평균 변화
    # ========================================================

    for window in MA_WINDOWS:

        ma = s.rolling(
            window,
            min_periods=window,
        ).mean()

        factors[
            f"{name}__ma_{window}m"
        ] = ma

        factors[
            f"{name}__dist_ma_{window}m"
        ] = distance_from_average(
            s,
            ma,
            kind,
        )

        for h in CHANGE_HORIZONS:

            factors[
                f"{name}__ma_{window}m_chg_{h}m"
            ] = change(
                ma,
                h,
                kind,
            )


    # ========================================================
    # 4. Z SCORE
    # ========================================================

    for window in Z_WINDOWS:

        factors[
            f"{name}__z_{window}m"
        ] = rolling_zscore(
            s,
            window,
        )


    # ========================================================
    # 5. EWMA
    # ========================================================

    for span in EWM_SPANS:

        ewma = s.ewm(
            span=span,
            adjust=False,
            min_periods=span,
        ).mean()

        factors[
            f"{name}__ewma_{span}m"
        ] = ewma

        factors[
            f"{name}__dist_ewma_{span}m"
        ] = distance_from_average(
            s,
            ewma,
            kind,
        )

        for h in CHANGE_HORIZONS:

            factors[
                f"{name}__ewma_{span}m_chg_{h}m"
            ] = change(
                ewma,
                h,
                kind,
            )


    # ========================================================
    # 6. 변화율 Volatility
    # ========================================================

    one_month_change = change(
        s,
        1,
        kind,
    )

    for window in VOL_WINDOWS:

        factors[
            f"{name}__vol_{window}m"
        ] = one_month_change.rolling(
            window,
            min_periods=window,
        ).std()


    # ========================================================
    # 7. Rolling Trend
    # ========================================================

    for window in SLOPE_WINDOWS:

        factors[
            f"{name}__slope_{window}m"
        ] = rolling_slope(
            s,
            window,
        )


    df = pd.DataFrame(
        factors
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df


def build_feature_matrix(
    panel,
    metadata,
):

    feature_frames = []

    for _, row in metadata.iterrows():

        name = row["name"]
        kind = row["kind"]

        if name not in panel.columns:
            continue

        print(
            f"[FACTOR] Factor 생성: "
            f"{name} ({kind})"
        )

        features = factor_factory(
            panel[name],
            name=name,
            kind=kind,
        )

        feature_frames.append(
            features
        )

    X = pd.concat(
        feature_frames,
        axis=1
    ).sort_index()

    X.index.name = "observation_date"

    return X


def build_cross_asset_features(monthly_panel):
    """Build predeclared cross-asset sensors from causally aligned raw legs."""
    required = {"usd_per_aud", "chf_per_usd"}
    missing = sorted(required.difference(monthly_panel.columns))
    if missing:
        raise KeyError(f"Cross-asset source series are missing: {missing}")

    factors = pd.DataFrame(index=monthly_panel.index)
    # DEXUSAL is USD per AUD and DEXSZUS is CHF per USD, so their product
    # is CHF per AUD: the conventional AUD/CHF cross-rate level.
    factors["aud_chf"] = (
        monthly_panel["usd_per_aud"] * monthly_panel["chf_per_usd"]
    )
    factors.index.name = "observation_date"
    metadata = pd.DataFrame(
        [
            {
                "feature": "aud_chf",
                "base": "aud_chf",
                "source": "FRED DEXUSAL × DEXSZUS",
                "group": "global_risk",
                "exactness": "exact_cross_rate",
                "description": "AUD/CHF (CHF per AUD) cross-rate",
                "availability_lag": 0,
                "point_in_time_rule": "same-month public daily FX observations",
                "model_eligible": True,
            }
        ]
    )
    return factors, metadata


def _monthly_stat(series, statistic, min_observations=10):

    def calculate(values):
        values = values.dropna()
        if len(values) < min_observations:
            return np.nan
        return statistic(values)

    return series.resample("ME").apply(calculate)


def _past_zscore(series, window=60, min_periods=24):
    """Rolling z-score using only information available through month t."""

    mean = series.rolling(
        window,
        min_periods=min_periods,
    ).mean()
    std = series.rolling(
        window,
        min_periods=min_periods,
    ).std()
    return (series - mean) / std.replace(0.0, np.nan)


def build_kospi200_market_features(market_daily):
    """Build a curated set of KOSPI200 (^KS200) OHLCV candidates.

    These are deliberately not sent through the full Factor Factory.  The
    goal is to add the economically interpretable indicators visible in the
    supplied candidate list without multiplying them into hundreds of close
    relatives and worsening the feature-search multiple-testing problem.

    `kospi200_index_daily_return_skew_1m_proxy` is a time-series proxy.  The
    photo's exact cross-sectional return skew requires constituent prices.
    """

    required = {"close", "high", "low", "open", "volume"}
    missing = required.difference(market_daily.columns)
    if missing:
        raise KeyError(
            "KOSPI 일별 데이터 필수 컬럼 누락: "
            + ", ".join(sorted(missing))
        )

    daily = market_daily.copy().sort_index()
    daily_return = daily["close"].pct_change(fill_method=None)
    monthly_close = daily["close"].resample("ME").last()
    monthly_return = monthly_close.pct_change(fill_method=None)

    skew_1m = _monthly_stat(
        daily_return,
        lambda values: values.skew(),
    )
    realized_vol_1m = _monthly_stat(
        daily_return,
        lambda values: values.std(ddof=1) * np.sqrt(252.0),
    )
    downside_vol_1m = _monthly_stat(
        daily_return,
        lambda values: (
            values.loc[values < 0].std(ddof=1) * np.sqrt(252.0)
            if (values < 0).sum() >= 2
            else np.nan
        ),
    )

    high_low_log_range = np.log(
        daily["high"] / daily["low"]
    ).replace([np.inf, -np.inf], np.nan)
    parkinson_vol_1m = _monthly_stat(
        high_low_log_range.pow(2),
        lambda values: np.sqrt(
            values.mean() * 252.0 / (4.0 * np.log(2.0))
        ),
    )

    monthly_volume = daily["volume"].resample("ME").sum(min_count=1)
    prior_volume_average = monthly_volume.shift(1).rolling(
        12,
        min_periods=6,
    ).mean()
    volume_ratio = monthly_volume / prior_volume_average

    momentum_3m = monthly_close.pct_change(3, fill_method=None)
    momentum_12m = monthly_close.pct_change(12, fill_method=None)
    rolling_peak_12m = monthly_close.rolling(12, min_periods=3).max()
    drawdown_12m = monthly_close / rolling_peak_12m - 1.0

    risk_components = pd.concat(
        [
            _past_zscore(momentum_3m).rename("momentum_3m"),
            _past_zscore(momentum_12m).rename("momentum_12m"),
            (-_past_zscore(realized_vol_1m)).rename("low_volatility"),
            _past_zscore(drawdown_12m).rename("shallow_drawdown"),
        ],
        axis=1,
    )
    risk_appetite_proxy = risk_components.mean(
        axis=1,
        skipna=False,
    )

    factors = pd.DataFrame(
        {
            "kospi200_return_1m": monthly_return,
            "kospi200_index_daily_return_skew_1m_proxy": skew_1m,
            "kospi200_trading_volume_ratio_12m": volume_ratio,
            "kospi200_realized_volatility_1m": realized_vol_1m,
            "kospi200_downside_volatility_1m": downside_vol_1m,
            "kospi200_parkinson_volatility_1m": parkinson_vol_1m,
            "kospi200_drawdown_12m": drawdown_12m,
            "kospi200_risk_appetite_price_proxy": risk_appetite_proxy,
        }
    ).replace([np.inf, -np.inf], np.nan)
    factors.index.name = "observation_date"

    descriptions = {
        "kospi200_return_1m": (
            "월말 KOSPI 종가의 1개월 수익률"
        ),
        "kospi200_index_daily_return_skew_1m_proxy": (
            "월중 KOSPI 일별수익률 왜도; 사진의 종목 횡단면 왜도에 "
            "대한 단일 지수 proxy"
        ),
        "kospi200_trading_volume_ratio_12m": (
            "당월 거래량 합계 / 직전 12개월 거래량 평균"
        ),
        "kospi200_realized_volatility_1m": (
            "월중 일별수익률 표준편차의 연율화 값"
        ),
        "kospi200_downside_volatility_1m": (
            "월중 음(-)의 일별수익률 변동성의 연율화 값"
        ),
        "kospi200_parkinson_volatility_1m": (
            "월중 고가/저가 범위를 이용한 Parkinson 연율 변동성"
        ),
        "kospi200_drawdown_12m": (
            "현재 월말 종가 / 최근 12개월 최고 월말 종가 - 1"
        ),
        "kospi200_risk_appetite_price_proxy": (
            "3·12개월 모멘텀, 낮은 실현변동성, 얕은 낙폭의 "
            "과거 60개월 rolling z-score 평균"
        ),
    }
    exactness = {
        feature: (
            "proxy"
            if feature.endswith("_proxy")
            else "direct"
        )
        for feature in factors.columns
    }
    metadata = pd.DataFrame(
        [
            {
                "feature": feature,
                "base": feature,
                "source": "Yahoo Finance ^KS200 daily OHLCV",
                "market": "KOSPI200",
                "ticker": "^KS200",
                "group": "market",
                "exactness": exactness[feature],
                "availability_lag": 0,
                "point_in_time_rule": "month-end data, next-month execution",
                "description": descriptions[feature],
                "first_date": factors[feature].first_valid_index(),
                "last_date": factors[feature].last_valid_index(),
                "observations": int(factors[feature].notna().sum()),
            }
            for feature in factors.columns
        ]
    )

    return factors, metadata


def build_market_index_features(
    market_daily,
    *,
    series_name="kospi200",
    market_name="KOSPI200",
    ticker="^KS200",
):
    """Build market-prefixed index OHLCV factors for any supported target.

    The established KOSPI200 calculations are reused exactly, then the factor
    identity and provenance are adapted to the selected market.  The daily
    return-skew variable remains explicitly labelled as an index time-series
    proxy, not constituent cross-sectional skew.
    """
    factors, metadata = build_kospi200_market_features(market_daily)
    rename = {
        column: column.replace("kospi200_", f"{series_name}_", 1)
        for column in factors.columns
    }
    factors = factors.rename(columns=rename)
    metadata = metadata.copy()
    metadata["feature"] = metadata["feature"].replace(rename)
    metadata["base"] = metadata["base"].replace(rename)
    metadata["source"] = f"Yahoo Finance {ticker} daily OHLCV"
    metadata["market"] = market_name
    metadata["ticker"] = ticker
    metadata["description"] = metadata["description"].astype(str).str.replace(
        "KOSPI", market_name, regex=False
    )
    monthly_close = market_daily["close"].sort_index().resample("ME").last()
    trend_factors = pd.DataFrame(
        {
            f"{series_name}_momentum_6m": monthly_close.pct_change(
                6, fill_method=None
            ),
            f"{series_name}_momentum_12m": monthly_close.pct_change(
                12, fill_method=None
            ),
            f"{series_name}_trend_10m": (
                monthly_close
                / monthly_close.rolling(10, min_periods=10).mean()
                - 1.0
            ),
        }
    )
    trend_factors.index.name = "observation_date"
    trend_descriptions = {
        f"{series_name}_momentum_6m": f"{market_name} six-month price momentum",
        f"{series_name}_momentum_12m": f"{market_name} twelve-month price momentum",
        f"{series_name}_trend_10m": (
            f"{market_name} month-end close relative to its trailing 10-month mean"
        ),
    }
    trend_metadata = pd.DataFrame(
        [
            {
                "feature": feature,
                "base": feature,
                "source": f"Yahoo Finance {ticker} daily OHLCV",
                "market": market_name,
                "ticker": ticker,
                "group": "market",
                "exactness": "direct",
                "availability_lag": 0,
                "point_in_time_rule": "month-end data, next-month execution",
                "description": description,
                "first_date": trend_factors[feature].first_valid_index(),
                "last_date": trend_factors[feature].last_valid_index(),
                "observations": int(trend_factors[feature].notna().sum()),
            }
            for feature, description in trend_descriptions.items()
        ]
    )
    factors = pd.concat([factors, trend_factors], axis=1)
    metadata = pd.concat([metadata, trend_metadata], ignore_index=True)
    return factors, metadata


def build_kospi_market_features(market_daily):
    """Backward-compatible alias for :func:`build_kospi200_market_features`."""
    return build_kospi200_market_features(market_daily)
