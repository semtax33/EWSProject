"""Causal KOSPI/cash allocation policies evaluated against KOSPI buy-and-hold."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics import performance_stats
from src.backtest import run_backtest


CANDIDATE_SPECS = {
    "absolute_momentum_3m_0_100": {
        "kind": "momentum",
        "lookback": 3,
        "risk_off_weight": 0.0,
        "risk_on_weight": 1.0,
    },
    "absolute_momentum_3m_20_100": {
        "kind": "momentum",
        "lookback": 3,
        "risk_off_weight": 0.2,
        "risk_on_weight": 1.0,
    },
    "absolute_momentum_12m_0_100": {
        "kind": "momentum",
        "lookback": 12,
        "risk_off_weight": 0.0,
        "risk_on_weight": 1.0,
    },
    "absolute_momentum_12m_20_100": {
        "kind": "momentum",
        "lookback": 12,
        "risk_off_weight": 0.2,
        "risk_on_weight": 1.0,
    },
    "sma_10m_0_100": {
        "kind": "sma",
        "lookback": 10,
        "risk_off_weight": 0.0,
        "risk_on_weight": 1.0,
    },
    "sma_12m_0_100": {
        "kind": "sma",
        "lookback": 12,
        "risk_off_weight": 0.0,
        "risk_on_weight": 1.0,
    },
}


def build_candidate_weights(market_price, specs=CANDIDATE_SPECS):
    """Build month-t targets; the backtest applies them only to month t+1."""
    price = market_price.sort_index().astype(float)
    weights = {}
    for name, spec in specs.items():
        lookback = int(spec["lookback"])
        if spec["kind"] == "momentum":
            indicator = price.pct_change(lookback, fill_method=None)
            risk_on = indicator > 0
            valid = indicator.notna()
        elif spec["kind"] == "sma":
            average = price.rolling(lookback, min_periods=lookback).mean()
            risk_on = price > average
            valid = average.notna() & price.notna()
        else:
            raise ValueError(f"Unknown allocation indicator: {spec['kind']}")
        weight = pd.Series(
            np.where(
                risk_on,
                float(spec["risk_on_weight"]),
                float(spec["risk_off_weight"]),
            ),
            index=price.index,
            dtype=float,
            name=name,
        ).where(valid)
        weights[name] = weight
    return weights


def build_candidate_backtests(
    market_price,
    cash_yield,
    *,
    transaction_cost_bps=10,
    specs=CANDIDATE_SPECS,
):
    weights = build_candidate_weights(market_price, specs=specs)
    return {
        name: run_backtest(
            market_price=market_price,
            ews=(weight * 100).rename("raw_ews"),
            target_stock_weight=weight,
            allocation_policy=name,
            cash_yield_annual_pct=cash_yield,
            transaction_cost_bps=transaction_cost_bps,
            verbose=False,
        )
        for name, weight in weights.items()
    }


def common_tradable_index(backtests):
    common = None
    for backtest in backtests.values():
        valid = backtest.dropna(
            subset=["strategy_return", "market_return", "executed_stock_weight"]
        ).index
        common = valid if common is None else common.intersection(valid)
    if common is None or common.empty:
        raise ValueError("No common tradable period across allocation candidates")
    return common


def evaluate_against_market(backtest, index, policy, period):
    data = backtest.loc[index].copy()
    active = data["strategy_return"] - data["market_return"]
    strategy = performance_stats(data["strategy_return"], data["cash_return"])
    benchmark = performance_stats(data["market_return"], data["cash_return"])
    relative_curve = (1 + data["strategy_return"]).cumprod() / (
        1 + data["market_return"]
    ).cumprod()
    active_std = active.std(ddof=1)
    rolling_strategy = (1 + data["strategy_return"]).rolling(36).apply(np.prod, raw=True)
    rolling_benchmark = (1 + data["market_return"]).rolling(36).apply(np.prod, raw=True)
    rolling_valid = rolling_strategy.notna() & rolling_benchmark.notna()
    rolling_win = (
        rolling_strategy.loc[rolling_valid] > rolling_benchmark.loc[rolling_valid]
    )
    return {
        "policy": policy,
        "period": period,
        "Start": data.index.min().date().isoformat(),
        "End": data.index.max().date().isoformat(),
        "Months": len(data),
        "average_stock_weight": float(data["executed_stock_weight"].mean()),
        "annual_turnover": float(data["turnover"].mean() * 12),
        "total_transaction_cost": float(data["transaction_cost"].sum()),
        "strategy_CAGR": strategy["CAGR"],
        "market_CAGR": benchmark["CAGR"],
        "CAGR_difference": strategy["CAGR"] - benchmark["CAGR"],
        "strategy_Sharpe": strategy["Sharpe"],
        "market_Sharpe": benchmark["Sharpe"],
        "Sharpe_difference": strategy["Sharpe"] - benchmark["Sharpe"],
        "strategy_MaxDrawdown": strategy["MaxDrawdown"],
        "market_MaxDrawdown": benchmark["MaxDrawdown"],
        "drawdown_improvement": strategy["MaxDrawdown"] - benchmark["MaxDrawdown"],
        "annualized_active_return": float(active.mean() * 12),
        "tracking_error": float(active_std * np.sqrt(12)),
        "information_ratio": (
            float(active.mean() / active_std * np.sqrt(12))
            if active_std > 0
            else np.nan
        ),
        "relative_terminal_wealth": float(relative_curve.iloc[-1] - 1),
        "rolling_36m_market_win_ratio": (
            float(rolling_win.mean()) if not rolling_win.empty else np.nan
        ),
    }


def evaluate_candidate_periods(
    backtests,
    *,
    research_end="2020-03-31",
    holdout_start="2020-04-30",
    holdout_end="2026-04-30",
):
    common = common_tradable_index(backtests)
    periods = {
        "pre2020_research": common[common <= pd.Timestamp(research_end)],
        "historical_holdout": common[
            (common >= pd.Timestamp(holdout_start))
            & (common <= pd.Timestamp(holdout_end))
        ],
        "full_history_diagnostic": common[common <= pd.Timestamp(holdout_end)],
    }
    rows = []
    for name, backtest in backtests.items():
        for period, index in periods.items():
            if len(index) < 12:
                continue
            rows.append(evaluate_against_market(backtest, index, name, period))
    return pd.DataFrame(rows), periods


def pre2020_fold_results(backtests, research_index, fold_months=24):
    rows = []
    for name, backtest in backtests.items():
        for offset in range(0, len(research_index), fold_months):
            index = research_index[offset : offset + fold_months]
            if len(index) < 12:
                continue
            row = evaluate_against_market(
                backtest, index, name, f"pre2020_fold_{offset // fold_months + 1}"
            )
            row["fold"] = offset // fold_months + 1
            rows.append(row)
    return pd.DataFrame(rows)


def select_benchmark_policy(comparison, folds):
    """Select only from pre-2020 results; never inspect holdout/full columns."""
    research = comparison.loc[comparison["period"].eq("pre2020_research")].copy()
    fold_summary = (
        folds.groupby("policy", as_index=False)
        .agg(
            median_fold_active_return=("annualized_active_return", "median"),
            positive_fold_ratio=("annualized_active_return", lambda x: float((x > 0).mean())),
            worst_fold_active_return=("annualized_active_return", "min"),
        )
    )
    decision = research.merge(fold_summary, on="policy", validate="one_to_one")
    decision["pre2020_gate"] = (
        (decision["CAGR_difference"] > 0)
        & (decision["relative_terminal_wealth"] > 0)
        & (decision["median_fold_active_return"] > 0)
        & (decision["drawdown_improvement"] >= 0.10)
    )
    eligible = decision.loc[decision["pre2020_gate"]]
    if eligible.empty:
        raise ValueError("No benchmark-oriented policy passed the pre-2020 gate")
    selected = eligible.sort_values(
        ["relative_terminal_wealth", "CAGR_difference", "policy"],
        ascending=[False, False, True],
    ).iloc[0]["policy"]
    decision["selected_pre2020"] = decision["policy"].eq(selected)
    decision["selection_data_end"] = "2020-03-31"
    decision["holdout_or_full_used_for_selection"] = False
    return str(selected), decision
