import numpy as np
import pandas as pd

from scipy.stats import spearmanr


# ============================================================
# RETURN IC
# ============================================================

def compute_return_ic(
    signal,
    future_return,
    rolling_window=36,
):

    df = pd.concat(
        [
            signal.rename("signal"),
            future_return.rename(
                "future_return"
            ),
        ],
        axis=1,
    ).dropna()

    if len(df) < 20:
        raise ValueError(
            "IC 계산 표본이 너무 적어."
        )

    # --------------------------------------------------------
    # Overall Rank IC
    # --------------------------------------------------------

    if (
        df["signal"].nunique() > 1
        and
        df["future_return"].nunique() > 1
    ):

        rank_ic = spearmanr(
            df["signal"],
            df["future_return"],
        ).statistic

    else:

        rank_ic = np.nan


    # --------------------------------------------------------
    # Pearson IC
    # --------------------------------------------------------

    pearson_ic = (
        df["signal"]
        .corr(
            df["future_return"]
        )
    )


    # --------------------------------------------------------
    # Rolling Rank IC
    # --------------------------------------------------------

    rolling_ic = pd.Series(
        np.nan,
        index=df.index,
        name="rolling_rank_ic",
    )

    for i in range(
        rolling_window - 1,
        len(df)
    ):

        window = df.iloc[
            i - rolling_window + 1:
            i + 1
        ]

        if (
            window["signal"].nunique()
            < 2
            or
            window[
                "future_return"
            ].nunique()
            < 2
        ):
            continue

        rolling_ic.iloc[i] = (
            spearmanr(
                window["signal"],
                window[
                    "future_return"
                ],
            ).statistic
        )

    valid_rolling = (
        rolling_ic.dropna()
    )

    summary = {
        "observations": len(df),

        "rank_ic": rank_ic,

        "pearson_ic":
            pearson_ic,

        "rolling_ic_mean":
            valid_rolling.mean(),

        "rolling_ic_median":
            valid_rolling.median(),

        "rolling_ic_std":
            valid_rolling.std(),

        "rolling_ic_positive_ratio":
            (
                valid_rolling > 0
            ).mean(),
    }

    return (
        summary,
        rolling_ic,
        df,
    )


# ============================================================
# ROLLING SHARPE
# ============================================================

def rolling_sharpe(
    returns,
    risk_free=None,
    window=36,
    periods_per_year=12,
):

    if risk_free is None:

        risk_free = pd.Series(
            0.0,
            index=returns.index,
        )

    df = pd.concat(
        [
            returns.rename("r"),
            risk_free.rename("rf"),
        ],
        axis=1,
    ).dropna()

    excess = (
        df["r"]
        - df["rf"]
    )

    mean = excess.rolling(
        window
    ).mean()

    std = excess.rolling(
        window
    ).std()

    sharpe = (
        mean / std
        * np.sqrt(
            periods_per_year
        )
    )

    sharpe.name = (
        "rolling_sharpe"
    )

    return sharpe


# ============================================================
# TOTAL PERFORMANCE
# ============================================================

def performance_stats(
    returns,
    risk_free=None,
    periods_per_year=12,
):

    if risk_free is None:

        risk_free = pd.Series(
            0.0,
            index=returns.index,
        )

    df = pd.concat(
        [
            returns.rename("r"),
            risk_free.rename("rf"),
        ],
        axis=1,
    ).dropna()

    if len(df) < 12:
        return {}

    r = df["r"]

    excess = (
        df["r"]
        - df["rf"]
    )

    # --------------------------------------------------------
    # CAGR
    # --------------------------------------------------------

    total_growth = (
        1 + r
    ).prod()

    years = (
        len(r)
        / periods_per_year
    )

    cagr = (
        total_growth
        ** (1 / years)
        - 1
    )


    # --------------------------------------------------------
    # Annualized Vol
    # --------------------------------------------------------

    annual_vol = (
        r.std()
        * np.sqrt(
            periods_per_year
        )
    )


    # --------------------------------------------------------
    # Sharpe
    # --------------------------------------------------------

    excess_std = (
        excess.std()
    )

    sharpe = (
        excess.mean()
        / excess_std
        * np.sqrt(
            periods_per_year
        )
        if excess_std > 0
        else np.nan
    )


    # --------------------------------------------------------
    # Sortino
    # --------------------------------------------------------

    downside = (
        excess[
            excess < 0
        ]
    )

    downside_std = (
        downside.std()
    )

    sortino = (
        excess.mean()
        / downside_std
        * np.sqrt(
            periods_per_year
        )
        if downside_std > 0
        else np.nan
    )


    # --------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------

    curve = (
        1 + r
    ).cumprod()

    peak = (
        curve.cummax()
    )

    drawdown = (
        curve / peak
        - 1
    )

    max_drawdown = (
        drawdown.min()
    )


    # --------------------------------------------------------
    # Calmar
    # --------------------------------------------------------

    calmar = (
        cagr
        / abs(max_drawdown)
        if max_drawdown < 0
        else np.nan
    )


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {
        "Months": len(r),

        "CAGR":
            cagr,

        "AnnualVol":
            annual_vol,

        "Sharpe":
            sharpe,

        "Sortino":
            sortino,

        "MaxDrawdown":
            max_drawdown,

        "Calmar":
            calmar,

        "MonthlyHitRate":
            (r > 0).mean(),
    }