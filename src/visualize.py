from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def plot_model_comparison(model_metrics, performance, save_path):
    """Compare signal and portfolio diagnostics on one historical window."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic = performance.loc[
        performance["strategy"].str.endswith(" Dynamic"),
        ["model", "Sharpe", "MaxDrawdown"],
    ]
    comparison = model_metrics[["model", "auc"]].merge(
        dynamic, on="model", how="inner", validate="one_to_one"
    )
    if comparison.empty:
        raise ValueError("No common model rows for comparison chart")

    colors = ["#2b8cbe", "#f28e2b", "#7b6fd0"][: len(comparison)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    panels = [
        ("auc", "AUC", 0.5),
        ("Sharpe", "Sharpe", 0.0),
        ("MaxDrawdown", "Max Drawdown", 0.0),
    ]
    for ax, (column, title, baseline) in zip(axes, panels):
        values = comparison[column].astype(float)
        if column == "MaxDrawdown":
            values = values * 100.0
        bars = ax.bar(comparison["model"], values, color=colors)
        ax.axhline(baseline, color="gray", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
        for bar, value in zip(bars, values):
            suffix = "%" if column == "MaxDrawdown" else ""
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}{suffix}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
            )
    fig.suptitle("Historical Research Holdout — Diagnostic Model Comparison")
    fig.tight_layout()
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_reliability(reliability, save_path, title="Probability reliability"):
    """Plot observed frequency against mean predicted probability."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    ax.plot(
        reliability["mean_probability"],
        reliability["observed_rate"],
        marker="o",
        linewidth=2,
        color="#3182bd",
        label="Observed",
    )
    for row in reliability.itertuples(index=False):
        ax.annotate(
            f"n={int(row.n)}",
            (row.mean_probability, row.observed_rate),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean probability", ylabel="Observed rate")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_dashboard(
    backtest,
    future_return,
    rolling_ic,
    rolling_sharpe,
    save_path,
    market_name="KOSPI200",
):

    save_path = Path(
        save_path
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = backtest.copy()

    valid = df[
        df["stock_weight"].notna()
    ].copy()

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(16, 14),
    )

    axes = axes.flatten()


    # ========================================================
    # 1. CUMULATIVE RETURN
    # ========================================================

    ax = axes[0]

    ax.plot(
        valid.index,
        valid["strategy_curve"],
        label="EWS Strategy",
        linewidth=2,
    )

    ax.plot(
        valid.index,
        valid["market_curve"],
        label=market_name,
        alpha=0.75,
    )

    ax.set_title(
        "Cumulative Return"
    )

    ax.set_ylabel(
        "Growth of 1"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )


    # ========================================================
    # 2. EWS + ALLOCATION
    # ========================================================

    ax = axes[1]

    ax.plot(
        valid.index,
        valid["ews"],
        label="EWS",
        color="black",
        linewidth=1.5,
    )

    ax.plot(
        valid.index,
        valid[
            "signal_stock_weight"
        ] * 100,
        label="Stock %",
        color="tab:blue",
    )

    ax.plot(
        valid.index,
        (
            1
            - valid[
                "signal_stock_weight"
            ]
        ) * 100,
        label="Cash %",
        color="tab:orange",
    )

    ax.axhline(
        50,
        color="gray",
        linestyle="--",
        alpha=0.5,
    )

    ax.set_ylim(
        0,
        100
    )

    ax.set_title(
        "EWS & Allocation"
    )

    ax.set_ylabel(
        "%"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )


    # ========================================================
    # 3. DRAWDOWN
    # ========================================================

    ax = axes[2]

    ax.fill_between(
        valid.index,
        valid[
            "strategy_drawdown"
        ] * 100,
        0,
        alpha=0.4,
        label="EWS Strategy",
    )

    ax.plot(
        valid.index,
        valid[
            "market_drawdown"
        ] * 100,
        color="red",
        alpha=0.8,
        label=market_name,
    )

    ax.set_title(
        "Drawdown"
    )

    ax.set_ylabel(
        "%"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )


    # ========================================================
    # 4. ROLLING IC
    # ========================================================

    ax = axes[3]

    ax.plot(
        rolling_ic.index,
        rolling_ic,
        color="purple",
    )

    ax.axhline(
        0,
        color="black",
        linewidth=1,
    )

    ax.axhline(
        0.1,
        color="green",
        linestyle="--",
        alpha=0.5,
    )

    ax.axhline(
        -0.1,
        color="red",
        linestyle="--",
        alpha=0.5,
    )

    ax.set_title(
        "Rolling Rank IC"
    )

    ax.grid(
        alpha=0.3
    )


    # ========================================================
    # 5. ROLLING SHARPE
    # ========================================================

    ax = axes[4]

    ax.plot(
        rolling_sharpe.index,
        rolling_sharpe,
        color="darkgreen",
    )

    ax.axhline(
        0,
        color="black",
    )

    ax.axhline(
        1,
        color="blue",
        linestyle="--",
        alpha=0.5,
    )

    ax.set_title(
        "Rolling Sharpe Ratio"
    )

    ax.grid(
        alpha=0.3
    )


    # ========================================================
    # 6. EWS vs FORWARD RETURN
    # ========================================================

    ax = axes[5]

    scatter_data = pd.concat(
        [
            df["ews"],
            future_return.rename(
                "future_return"
            ),
        ],
        axis=1,
    ).dropna()

    x = (
        scatter_data["ews"]
        .values
    )

    y = (
        scatter_data[
            "future_return"
        ].values
        * 100
    )

    ax.scatter(
        x,
        y,
        alpha=0.5,
        s=25,
    )

    if (
        len(x) >= 10
        and
        np.std(x) > 0
    ):

        slope, intercept = (
            np.polyfit(
                x,
                y,
                1
            )
        )

        x_line = np.linspace(
            x.min(),
            x.max(),
            100,
        )

        ax.plot(
            x_line,
            (
                slope
                * x_line
                + intercept
            ),
            color="red",
            linewidth=2,
        )

    ax.axhline(
        0,
        color="black",
        linewidth=1,
    )

    ax.set_xlabel(
        "EWS"
    )

    ax.set_ylabel(
        "Forward Return (%)"
    )

    ax.set_title(
        "EWS vs Forward Return"
    )

    ax.grid(
        alpha=0.3
    )


    # ========================================================
    # FINISH
    # ========================================================

    fig.suptitle(
        "Macro EWS Model Dashboard",
        fontsize=18,
        fontweight="bold",
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97
        ]
    )

    fig.savefig(
        save_path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(fig)

def plot_latest_allocation(
    stock_weight,
    cash_weight,
    ews,
    date,
    save_path,
):

    save_path = Path(
        save_path
    )

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    values = [
        stock_weight,
        cash_weight,
    ]

    labels = [
        f"Stock\n{stock_weight:.1%}",
        f"Cash\n{cash_weight:.1%}",
    ]

    colors = [
        "#3182bd",
        "#bdbdbd",
    ]

    ax.pie(
        values,
        labels=labels,
        colors=colors,
        startangle=90,
        autopct=None,
        wedgeprops={
            "width": 0.45,
            "edgecolor": "white",
        },
    )

    ax.text(
        0,
        0.08,
        f"EWS\n{ews:.1f}",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
    )

    ax.text(
        0,
        -0.25,
        str(date),
        ha="center",
        va="center",
        fontsize=10,
        color="gray",
    )

    ax.set_title(
        "Current Asset Allocation",
        fontsize=16,
    )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(fig)
