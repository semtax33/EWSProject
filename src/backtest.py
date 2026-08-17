import numpy as np
import pandas as pd

from src.position_sizing import target_weight_from_ews


# ============================================================
# CASH YIELD
# ============================================================

def annual_yield_to_monthly_return(
    annual_yield_pct,
    convention="simple_divide_12",
):
    """
    연율 % 금리를 단순 월 수익률 proxy로 변환.

    예:
    연 4.8% -> 월 약 0.4%

    annual_yield_pct가
    4.8이라면 4.8%를 의미.
    """

    annual_decimal = annual_yield_pct / 100.0
    if convention == "simple_divide_12":
        return annual_decimal / 12.0
    if convention == "effective_annual_compound":
        return (1.0 + annual_decimal).pow(1.0 / 12.0) - 1.0
    raise ValueError(f"Unknown cash-return convention: {convention}")


# ============================================================
# EWS -> STOCK WEIGHT
# ============================================================

def score_to_stock_weight(
    ews,
    min_weight=0.20,
    max_weight=0.80,
):
    """
    EWS 0~100을 주식 비중 0~1로 변환.

    예:
        EWS = 72
        -> 기본 주식비중 72%

    단:
        min_weight = 0.20
        max_weight = 0.80

    이면 실제 비중은 20~80% 사이로 제한.
    """

    weight = (
        ews
        / 100.0
    )

    return weight.clip(
        lower=min_weight,
        upper=max_weight,
    )


# ============================================================
# CURVE + DRAWDOWN HELPER
# ============================================================

def _add_curve_and_drawdown(
    df,
    return_col,
    valid_mask,
    curve_col,
    drawdown_col,
):
    """
    특정 수익률 컬럼에서
    누적수익률과 Drawdown을 생성.
    """

    returns = (
        df.loc[
            valid_mask,
            return_col
        ]
        .fillna(0.0)
    )

    curve = (
        1.0
        + returns
    ).cumprod()

    df.loc[
        valid_mask,
        curve_col
    ] = curve

    running_peak = (
        curve.cummax()
    )

    drawdown = (
        curve
        / running_peak
        - 1.0
    )

    df.loc[
        valid_mask,
        drawdown_col
    ] = drawdown


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    market_price,
    ews=None,
    cash_yield_annual_pct=None,
    min_stock_weight=0.20,
    max_stock_weight=0.80,
    transaction_cost_bps=10,
    benchmark_stock_weight=0.50,
    allocation_policy="linear",
    target_stock_weight=None,
    allocation_kwargs=None,
    cash_return_convention="simple_divide_12",
    verbose=True,
    market_name="KOSPI200",
):
    """
    EWS 기반 Tactical Asset Allocation 백테스트.

    비교 대상:
        1. Dynamic EWS
        2. Static 50/50
        3. Static Same Exposure
        4. KOSPI 100%

    Parameters
    ----------
    market_price : pd.Series
        월말 KOSPI 가격

    ews : pd.Series
        0~100 범위의 EWS

    cash_yield_annual_pct : pd.Series or None
        연율 % 현금금리.
        예: DGS3MO

    min_stock_weight : float
        최소 주식 비중

    max_stock_weight : float
        최대 주식 비중

    transaction_cost_bps : float
        EWS 전략의 거래비용.
        10 = 10bp = 0.10%

    benchmark_stock_weight : float
        정적 benchmark 주식비중.
        기본값 0.50
    """

    # ========================================================
    # 0. INDEX 정리
    # ========================================================

    market_price = (
        market_price
        .copy()
        .sort_index()
    )

    if ews is None and target_stock_weight is None:
        raise ValueError("ews or target_stock_weight must be provided")

    if ews is None:
        ews = pd.Series(
            np.nan,
            index=target_stock_weight.index,
            name="raw_ews",
        )
    else:
        ews = (
            ews
            .copy()
            .sort_index()
        )

    # 혹시 중복 날짜가 있으면 마지막 값 사용
    market_price = market_price[
        ~market_price.index.duplicated(
            keep="last"
        )
    ]

    ews = ews[
        ~ews.index.duplicated(
            keep="last"
        )
    ]

    if target_stock_weight is not None:
        target_stock_weight = target_stock_weight.copy().sort_index()
        target_stock_weight = target_stock_weight[
            ~target_stock_weight.index.duplicated(keep="last")
        ]
        invalid_weight = target_stock_weight.dropna()
        invalid_weight = invalid_weight[
            (invalid_weight < 0.0) | (invalid_weight > 1.0)
        ]
        if not invalid_weight.empty:
            raise ValueError("target_stock_weight must stay between 0 and 1")


    # ========================================================
    # 1. 기본 DataFrame
    # ========================================================

    df = pd.concat(
        [
            market_price.rename(
                "market_price"
            ),
            ews.rename(
                "raw_ews"
            ),
        ],
        axis=1,
    ).sort_index()

    # Backward-compatible alias for result readers created before the raw
    # model score and allocation weight were separated.
    df["ews"] = df["raw_ews"]


    # ========================================================
    # 2. KOSPI 월 수익률
    # ========================================================

    df["market_return"] = (
        df["market_price"]
        .pct_change(
            fill_method=None
        )
    )


    # ========================================================
    # 3. 현금 수익률
    # ========================================================

    if cash_yield_annual_pct is not None:

        cash_yield = (
            cash_yield_annual_pct
            .copy()
            .sort_index()
        )

        cash_yield = cash_yield[
            ~cash_yield.index.duplicated(
                keep="last"
            )
        ]

        cash_yield = (
            cash_yield
            .reindex(
                df.index
            )
            .ffill()
        )

        df[
            "cash_yield_annual_pct"
        ] = cash_yield

        monthly_cash_return = (
            annual_yield_to_monthly_return(
                cash_yield,
                convention=cash_return_convention,
            )
        )

        # ----------------------------------------------
        # t-1월 말에 알고 있던 금리를
        # t월 현금 수익률로 사용
        # ----------------------------------------------

        df["cash_return"] = (
            monthly_cash_return
            .shift(1)
            .fillna(0.0)
        )

    else:

        df[
            "cash_yield_annual_pct"
        ] = np.nan

        df[
            "cash_return"
        ] = 0.0


    # ========================================================
    # 4. EWS -> 원하는 주식 비중
    # ========================================================

    if target_stock_weight is None:
        policy_options = dict(allocation_kwargs or {})
        policy_options.setdefault("min_weight", min_stock_weight)
        policy_options.setdefault("max_weight", max_stock_weight)
        desired_stock_weight = target_weight_from_ews(
            df["raw_ews"],
            policy=allocation_policy,
            **policy_options,
        )
    else:
        desired_stock_weight = target_stock_weight.reindex(df.index)

    df["allocation_policy"] = allocation_policy
    df["target_stock_weight"] = desired_stock_weight

    df[
        "signal_stock_weight"
    ] = desired_stock_weight

    df[
        "signal_cash_weight"
    ] = (
        1.0
        - desired_stock_weight
    )


    # ========================================================
    # 5. 미래참조 방지
    #
    # t월 말 EWS
    #     ↓
    # t+1월 투자비중
    # ========================================================

    df["stock_weight"] = (
        desired_stock_weight
        .shift(1)
    )

    df["executed_stock_weight"] = df["stock_weight"]

    df["cash_weight"] = (
        1.0
        - df["stock_weight"]
    )


    # ========================================================
    # 6. 유효 백테스트 구간
    # ========================================================

    valid = (
        df["stock_weight"].notna()
        &
        df["market_return"].notna()
    )

    if valid.sum() == 0:

        raise ValueError(
            "백테스트 가능한 기간이 없어. "
            "EWS 날짜와 KOSPI 날짜를 확인해."
        )


    # ========================================================
    # 7. Turnover
    # ========================================================

    df["turnover"] = (
        df["stock_weight"]
        .diff()
        .abs()
    )

    first_valid = (
        df.index[
            valid
        ][0]
    )

    # ----------------------------------------------
    # 최초에는 현금 100%에서 시작했다고 가정.
    #
    # 예:
    # 첫 주식비중이 60%라면
    # 60%만큼 주식을 매수했다고 처리.
    # ----------------------------------------------

    df.loc[
        first_valid,
        "turnover"
    ] = df.loc[
        first_valid,
        "stock_weight"
    ]


    # ========================================================
    # 8. Transaction Cost
    # ========================================================

    cost_rate = (
        transaction_cost_bps
        / 10000.0
    )

    df[
        "transaction_cost"
    ] = (
        df["turnover"]
        .fillna(0.0)
        * cost_rate
    )


    # ========================================================
    # 9. Dynamic EWS Return
    # ========================================================

    df["strategy_return"] = (

        df["stock_weight"]
        * df["market_return"]

        +

        df["cash_weight"]
        * df["cash_return"]

        -

        df[
            "transaction_cost"
        ]
    )


    # ========================================================
    # 10. STATIC 50/50 BENCHMARK
    #
    # 기본값:
    # KOSPI 50%
    # Cash  50%
    #
    # passive benchmark라
    # 여기서는 거래비용을 부과하지 않음.
    # 즉 EWS 입장에서는 오히려 더 빡센 비교.
    # ========================================================

    benchmark_stock_weight = float(
        benchmark_stock_weight
    )

    if not (
        0.0
        <= benchmark_stock_weight
        <= 1.0
    ):

        raise ValueError(
            "benchmark_stock_weight는 "
            "0~1 사이여야 해."
        )

    benchmark_cash_weight = (
        1.0
        - benchmark_stock_weight
    )

    df[
        "benchmark_50_50_stock_weight"
    ] = benchmark_stock_weight

    df[
        "benchmark_50_50_cash_weight"
    ] = benchmark_cash_weight

    df[
        "benchmark_50_50_return"
    ] = (

        benchmark_stock_weight
        * df["market_return"]

        +

        benchmark_cash_weight
        * df["cash_return"]
    )


    # ========================================================
    # 11. SAME EXPOSURE STATIC BENCHMARK
    #
    # EWS가 Test 기간 동안 평균적으로
    # 얼마의 주식을 들고 있었는가?
    #
    # 예:
    # Dynamic EWS 평균 주식비중 = 53%
    #
    # 그러면:
    # Static Same Exposure =
    # KOSPI 53% + Cash 47%
    #
    # 이게 "Market Timing 자체의 가치"를 측정하는
    # 가장 중요한 Benchmark.
    # ========================================================

    avg_stock_weight = (
        df.loc[
            valid,
            "stock_weight"
        ].mean()
    )

    avg_cash_weight = (
        1.0
        - avg_stock_weight
    )

    df[
        "average_stock_weight"
    ] = avg_stock_weight

    df[
        "average_cash_weight"
    ] = avg_cash_weight

    df[
        "same_exposure_return"
    ] = (

        avg_stock_weight
        * df["market_return"]

        +

        avg_cash_weight
        * df["cash_return"]
    )


    # ========================================================
    # 12. ACTIVE RETURN
    #
    # EWS가 정적 benchmark를
    # 월별로 얼마나 이겼는지 확인.
    # ========================================================

    df[
        "active_return_vs_50_50"
    ] = (
        df["strategy_return"]
        - df[
            "benchmark_50_50_return"
        ]
    )

    df[
        "active_return_vs_same_exposure"
    ] = (
        df["strategy_return"]
        - df[
            "same_exposure_return"
        ]
    )

    df[
        "active_return_vs_kospi"
    ] = (
        df["strategy_return"]
        - df["market_return"]
    )
    df["active_return_vs_kospi200"] = df["active_return_vs_kospi"]


    # ========================================================
    # 13. CUMULATIVE CURVES + DRAWDOWNS
    # ========================================================

    # Dynamic EWS
    _add_curve_and_drawdown(
        df=df,
        return_col=(
            "strategy_return"
        ),
        valid_mask=valid,
        curve_col=(
            "strategy_curve"
        ),
        drawdown_col=(
            "strategy_drawdown"
        ),
    )

    # KOSPI 100%
    _add_curve_and_drawdown(
        df=df,
        return_col=(
            "market_return"
        ),
        valid_mask=valid,
        curve_col=(
            "market_curve"
        ),
        drawdown_col=(
            "market_drawdown"
        ),
    )

    # Static 50/50
    _add_curve_and_drawdown(
        df=df,
        return_col=(
            "benchmark_50_50_return"
        ),
        valid_mask=valid,
        curve_col=(
            "benchmark_50_50_curve"
        ),
        drawdown_col=(
            "benchmark_50_50_drawdown"
        ),
    )

    # Same Exposure Static
    _add_curve_and_drawdown(
        df=df,
        return_col=(
            "same_exposure_return"
        ),
        valid_mask=valid,
        curve_col=(
            "same_exposure_curve"
        ),
        drawdown_col=(
            "same_exposure_drawdown"
        ),
    )


    # ========================================================
    # 14. 누적 Active Return Curve
    #
    # 단순 확인용.
    # 1보다 높아지면 Dynamic EWS가
    # 해당 정적전략 대비 누적으로 우세.
    # ========================================================

    df.loc[
        valid,
        "relative_curve_vs_50_50"
    ] = (
        (
            1.0
            + df.loc[
                valid,
                "strategy_return"
            ].fillna(0.0)
        ).cumprod()

        /

        (
            1.0
            + df.loc[
                valid,
                "benchmark_50_50_return"
            ].fillna(0.0)
        ).cumprod()
    )

    df.loc[
        valid,
        "relative_curve_vs_same_exposure"
    ] = (
        (
            1.0
            + df.loc[
                valid,
                "strategy_return"
            ].fillna(0.0)
        ).cumprod()

        /

        (
            1.0
            + df.loc[
                valid,
                "same_exposure_return"
            ].fillna(0.0)
        ).cumprod()
    )


    # ========================================================
    # 15. DEBUG / SANITY INFO
    # ========================================================

    if not verbose:
        return df

    print()
    print("=" * 65)
    print("📊 BACKTEST SETUP")
    print("=" * 65)

    print(
        f"백테스트 기간       : "
        f"{df.index[valid][0].date()} "
        f"~ "
        f"{df.index[valid][-1].date()}"
    )

    print(
        f"백테스트 개월       : "
        f"{valid.sum()}"
    )

    print(
        f"EWS 평균 주식비중   : "
        f"{avg_stock_weight:.1%}"
    )

    print(
        f"EWS 평균 현금비중   : "
        f"{avg_cash_weight:.1%}"
    )

    print(
        f"Static Benchmark    : "
        f"{market_name} "
        f"{benchmark_stock_weight:.0%} "
        f"/ Cash "
        f"{benchmark_cash_weight:.0%}"
    )

    print(
        f"거래비용            : "
        f"{transaction_cost_bps:.1f} bps"
    )

    print("=" * 65)

    return df
