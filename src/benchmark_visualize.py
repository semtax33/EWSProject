"""Headless charts for the benchmark-allocation research outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STRATEGY_COLOR = "#2563eb"
MARKET_COLOR = "#6b7280"
ACTIVE_COLOR = "#059669"
NEGATIVE_COLOR = "#dc2626"
SELECTED_COLOR = "#f59e0b"
HOLDOUT_START = pd.Timestamp("2020-04-30")


def _save(fig, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _mark_holdout(ax, start=HOLDOUT_START):
    ax.axvline(start, color="#7c3aed", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(
        start,
        0.98,
        "  post-2020 diagnostic",
        transform=ax.get_xaxis_transform(),
        color="#7c3aed",
        fontsize=8,
        va="top",
    )


def _report_row(report):
    if isinstance(report, pd.DataFrame):
        if report.empty:
            raise ValueError("report must contain one row")
        return report.iloc[0]
    return report


def plot_benchmark_allocation_dashboard(backtest, report, save_path):
    """Plot wealth, drawdown, allocation and relative wealth for the selected policy."""
    valid = backtest.dropna(
        subset=["strategy_return", "market_return", "executed_stock_weight"]
    ).copy()
    if valid.empty:
        raise ValueError("backtest has no valid observations to plot")
    valid = valid.loc[:"2026-04-30"]
    row = _report_row(report)

    strategy_curve = (1 + valid["strategy_return"]).cumprod()
    market_curve = (1 + valid["market_return"]).cumprod()
    strategy_drawdown = strategy_curve / strategy_curve.cummax() - 1
    market_drawdown = market_curve / market_curve.cummax() - 1
    relative_curve = strategy_curve / market_curve

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)

    ax = axes[0, 0]
    ax.plot(strategy_curve.index, strategy_curve, color=STRATEGY_COLOR, lw=2.2, label="Selected strategy")
    ax.plot(market_curve.index, market_curve, color=MARKET_COLOR, lw=1.7, label="KOSPI200 buy & hold")
    ax.set_yscale("log")
    ax.set_title("Growth of 1 (log scale)")
    ax.set_ylabel("Portfolio value")
    ax.legend(loc="upper left")
    summary = (
        f"CAGR  {row['strategy_CAGR']:.1%} vs {row['market_CAGR']:.1%}\n"
        f"Sharpe  {row['strategy_Sharpe']:.2f} vs {row['market_Sharpe']:.2f}\n"
        f"MDD  {row['strategy_MaxDrawdown']:.1%} vs {row['market_MaxDrawdown']:.1%}"
    )
    ax.text(
        0.02,
        0.70,
        summary,
        transform=ax.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.45", "fc": "white", "ec": "#d1d5db", "alpha": 0.9},
    )
    _mark_holdout(ax)

    ax = axes[0, 1]
    ax.fill_between(valid.index, strategy_drawdown * 100, 0, color=STRATEGY_COLOR, alpha=0.28, label="Selected strategy")
    ax.plot(valid.index, market_drawdown * 100, color=MARKET_COLOR, lw=1.5, label="KOSPI200")
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left")
    _mark_holdout(ax)

    ax = axes[1, 0]
    ax.step(valid.index, valid["executed_stock_weight"] * 100, where="post", color=SELECTED_COLOR, lw=1.5)
    ax.fill_between(valid.index, valid["executed_stock_weight"] * 100, step="post", color=SELECTED_COLOR, alpha=0.16)
    ax.set_ylim(-3, 103)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title("Executed KOSPI200 allocation (one-month lag)")
    ax.set_ylabel("Stock weight (%)")
    _mark_holdout(ax)

    ax = axes[1, 1]
    ax.plot(relative_curve.index, relative_curve, color=ACTIVE_COLOR, lw=2)
    ax.axhline(1, color="black", lw=1, alpha=0.7)
    ax.fill_between(relative_curve.index, relative_curve, 1, where=relative_curve.ge(1), color=ACTIVE_COLOR, alpha=0.15)
    ax.fill_between(relative_curve.index, relative_curve, 1, where=relative_curve.lt(1), color=NEGATIVE_COLOR, alpha=0.12)
    ax.set_title("Relative wealth: strategy / KOSPI200")
    ax.set_ylabel("Relative wealth")
    _mark_holdout(ax)

    for ax in axes.flat:
        ax.grid(alpha=0.22)
        ax.margins(x=0.01)
    policy = _short_policy_name(row.get("policy", "selected policy"))
    fig.suptitle(
        f"KOSPI200 / Cash Allocation Dashboard\n{policy} | 10 bp turnover cost | decision at t, execution at t+1",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, save_path)


def plot_outperformance_diagnostics(rolling, calendar, regime, save_path):
    """Plot rolling, calendar-year and market-regime active performance."""
    rolling = rolling.copy()
    rolling["window_end"] = pd.to_datetime(rolling["window_end"])
    calendar = calendar.copy()
    regime = regime.copy()

    fig, axes = plt.subplots(3, 1, figsize=(15, 13))

    ax = axes[0]
    for window, color in zip((12, 36, 60), ("#0ea5e9", STRATEGY_COLOR, "#7c3aed")):
        data = rolling.loc[rolling["window_months"].eq(window)]
        ax.plot(data["window_end"], data["active_total_return"] * 100, lw=1.7, color=color, label=f"{window} months")
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Rolling total-return advantage over KOSPI200")
    ax.set_ylabel("Active return (pp)")
    ax.legend(ncol=3)
    _mark_holdout(ax)

    ax = axes[1]
    colors = np.where(calendar["active_return"].ge(0), ACTIVE_COLOR, NEGATIVE_COLOR)
    bars = ax.bar(calendar["year"].astype(str), calendar["active_return"] * 100, color=colors, alpha=0.85)
    incomplete = calendar["months"].lt(12)
    for bar, partial in zip(bars, incomplete):
        if partial:
            bar.set_hatch("//")
            bar.set_alpha(0.5)
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Calendar-year active return (hatched years are partial)")
    ax.set_ylabel("Active return (pp)")
    ax.tick_params(axis="x", rotation=55, labelsize=8)

    ax = axes[2]
    labels = [value.replace("_", " ").title() for value in regime["regime"]]
    values = regime["annualized_active_return"] * 100
    colors = np.where(values.ge(0), ACTIVE_COLOR, NEGATIVE_COLOR)
    bars = ax.bar(labels, values, color=colors, alpha=0.85)
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Annualized active return by market regime")
    ax.set_ylabel("Active return (pp/year)")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + (0.8 if value >= 0 else -1.5), f"{value:.1f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    ax.margins(y=0.16)

    for ax in axes:
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Historical Outperformance Diagnostics", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, save_path)


def _short_policy_name(policy):
    replacements = {
        "absolute_momentum_3m_0_100": "3M momentum 0/100",
        "absolute_momentum_3m_20_100": "3M momentum 20/100",
        "absolute_momentum_12m_0_100": "12M momentum 0/100",
        "absolute_momentum_12m_20_100": "12M momentum 20/100",
        "sma_10m_0_100": "10M SMA 0/100",
        "sma_12m_0_100": "12M SMA 0/100",
    }
    return replacements.get(policy, policy)


def plot_candidate_selection(comparison, selected_policy, save_path):
    """Visualize metrics actually available to the pre-2020 policy selection."""
    data = comparison.loc[comparison["period"].eq("pre2020_research")].copy()
    if data.empty:
        raise ValueError("comparison has no pre2020_research rows")
    data = data.sort_values("CAGR_difference")
    labels = [_short_policy_name(value) for value in data["policy"]]
    colors = [SELECTED_COLOR if value == selected_policy else "#94a3b8" for value in data["policy"]]
    metrics = (
        ("CAGR_difference", "CAGR advantage", "Percentage points"),
        ("drawdown_improvement", "Maximum-drawdown improvement", "Percentage points"),
        ("rolling_36m_market_win_ratio", "Rolling 36M market win ratio", "Percent"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
    for ax, (column, title, xlabel) in zip(axes, metrics):
        values = data[column] * 100
        bars = ax.barh(labels, values, color=colors, alpha=0.9)
        ax.axvline(0, color="black", lw=1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.22)
        right = max(float(values.max()) * 1.16, 1.0)
        left = min(0.0, float(values.min()) * 1.16)
        ax.set_xlim(left, right)
        for bar, value in zip(bars, values):
            offset = (right - left) * 0.015
            ax.text(value + (offset if value >= 0 else -offset), bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", ha="left" if value >= 0 else "right", fontsize=8, clip_on=True)
    fig.suptitle(
        "Candidate Selection Using Pre-2020 Data Only\norange = selected; 2020-04 onward was not used for selection",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    _save(fig, save_path)


def plot_baseline_comparison(artifact, same_period, save_path):
    """Plot the like-for-like baseline/current diagnostic comparison."""
    artifact = artifact.set_index("metric")
    same_period = same_period.copy()
    labels = [value.replace("_", " ").replace("20260812", "") for value in same_period["specification"]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    signal_metrics = ["holdout_auc", "holdout_rank_score"]
    x = np.arange(len(signal_metrics))
    width = 0.34
    ax.bar(x - width / 2, artifact.loc[signal_metrics, "baseline"], width, label="Baseline", color="#94a3b8")
    ax.bar(x + width / 2, artifact.loc[signal_metrics, "current"], width, label="Current", color=STRATEGY_COLOR)
    ax.set_xticks(x, ["Holdout AUC", "Holdout rank score"])
    ax.set_ylim(0.5, 0.85)
    ax.set_title("Signal diagnostics")
    ax.legend()

    ax = axes[0, 1]
    x = np.arange(len(labels))
    ax.bar(x - width / 2, same_period["dynamic_Sharpe"], width, label="Dynamic", color=STRATEGY_COLOR)
    ax.bar(x + width / 2, same_period["same_exposure_Sharpe"], width, label="Same exposure", color="#94a3b8")
    ax.set_xticks(x, labels)
    ax.set_title("Same-period Sharpe (2020-05 to 2026-05)")
    ax.legend()

    ax = axes[1, 0]
    active = same_period["active_Sharpe_delta"]
    ax.bar(labels, active, color=np.where(active.ge(0), ACTIVE_COLOR, NEGATIVE_COLOR))
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Dynamic minus same-exposure Sharpe")
    ax.set_ylabel("Sharpe difference")

    ax = axes[1, 1]
    x = np.arange(len(labels))
    ax.bar(x - width / 2, same_period["dynamic_CAGR"] * 100, width, label="Dynamic", color=STRATEGY_COLOR)
    ax.bar(x + width / 2, same_period["same_exposure_CAGR"] * 100, width, label="Same exposure", color="#94a3b8")
    ax.set_xticks(x, labels)
    ax.set_title("Same-period CAGR")
    ax.set_ylabel("Percent")
    ax.legend()

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.22)
        ax.tick_params(axis="x", rotation=8)
    fig.suptitle(
        "Baseline vs Current: Like-for-Like Diagnostics\nThese charts do not compare holdout Sharpe with nested pre-2020 AUC",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, save_path)
