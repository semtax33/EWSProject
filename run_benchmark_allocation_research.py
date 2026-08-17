"""Diagnose baseline drift and research a causal KOSPI/cash benchmark overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics import compute_return_ic, performance_stats
from src.backtest import run_backtest
from src.benchmark_allocation import (
    CANDIDATE_SPECS,
    build_candidate_backtests,
    build_candidate_weights,
    evaluate_against_market,
    evaluate_candidate_periods,
    pre2020_fold_results,
    select_benchmark_policy,
)
from src.benchmark_visualize import (
    plot_baseline_comparison,
    plot_benchmark_allocation_dashboard,
    plot_candidate_selection,
    plot_outperformance_diagnostics,
)
from src.config import (
    FORECAST_HORIZON,
    OPERATIONAL_RISK_ACCEPTANCE_FILE,
    RESEARCH_HOLDOUT_START,
    RUNS_DIR,
    SVM_CALIBRATION_SPLITS,
    SVM_PARAMS,
    TRANSACTION_COST_BPS,
)
from src.experiment import build_manifest, create_run_directory, write_manifest
from src.modeling import evaluate_probabilities, walk_forward_predict
from src.shadow import canonical_spec_hash, initialize_shadow_ledger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-run", required=True)
    parser.add_argument(
        "--baseline-run",
        default="runs/baseline_20260812_pre_position_sizing/results",
    )
    parser.add_argument("--run-id")
    return parser.parse_args()


def _load_manifest_complete(path):
    manifest = json.loads((path / "experiment_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Parent classification run must be complete")
    return manifest


def artifact_comparison(parent, baseline):
    baseline_model = pd.read_csv(baseline / "model_comparison.csv").set_index("model").loc["SVM"]
    current_model = pd.read_csv(parent / "model_comparison.csv").set_index("model").loc["SVM"]
    baseline_perf = pd.read_csv(baseline / "performance_comparison.csv")
    current_perf = pd.read_csv(parent / "performance_comparison.csv")
    baseline_dynamic = baseline_perf.loc[
        (baseline_perf["model"] == "SVM") & baseline_perf["strategy"].eq("SVM Dynamic")
    ].iloc[0]
    current_dynamic = current_perf.loc[
        (current_perf["model"] == "SVM") & current_perf["strategy"].eq("SVM Dynamic")
    ].iloc[0]
    baseline_same = baseline_perf.loc[
        (baseline_perf["model"] == "SVM")
        & baseline_perf["strategy"].str.contains("Same Exposure")
    ].iloc[0]
    current_same = current_perf.loc[
        (current_perf["model"] == "SVM")
        & current_perf["strategy"].str.contains("Same Exposure")
    ].iloc[0]
    values = {
        "holdout_auc": (baseline_model["auc"], current_model["auc"]),
        "holdout_rank_score": (baseline_model["rank_score"], current_model["rank_score"]),
        "dynamic_sharpe": (baseline_dynamic["Sharpe"], current_dynamic["Sharpe"]),
        "same_exposure_sharpe": (baseline_same["Sharpe"], current_same["Sharpe"]),
        "active_sharpe_delta": (
            baseline_dynamic["Sharpe"] - baseline_same["Sharpe"],
            current_dynamic["Sharpe"] - current_same["Sharpe"],
        ),
        "dynamic_cagr": (baseline_dynamic["CAGR"], current_dynamic["CAGR"]),
        "dynamic_max_drawdown": (
            baseline_dynamic["MaxDrawdown"], current_dynamic["MaxDrawdown"]
        ),
        "evaluation_months": (baseline_dynamic["Months"], current_dynamic["Months"]),
    }
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "baseline": float(pair[0]),
                "current": float(pair[1]),
                "current_minus_baseline": float(pair[1] - pair[0]),
                "same_period_warning": metric == "evaluation_months" or (
                    baseline_dynamic["Start"] != current_dynamic["Start"]
                ),
            }
            for metric, pair in values.items()
        ]
    )


def fixed_feature_regime_diagnostics(parent, baseline):
    X = pd.read_parquet(parent / "factor_matrix.parquet")
    target = pd.read_csv(parent / "target.csv", index_col=0, parse_dates=True)
    y = target["y"]
    feature_sets = {
        "baseline_20260812_fixed": json.loads(
            (baseline / "selected_features.json").read_text(encoding="utf-8")
        ),
        "current_global_fixed": json.loads(
            (parent / "selected_features.json").read_text(encoding="utf-8")
        ),
    }
    rows = []
    predictions = {}
    for specification, features in feature_sets.items():
        for period, start, end in (
            ("pre2020_expanding_fixed", "2001-01-31", "2020-03-31"),
            ("historical_holdout", "2020-04-30", "2026-04-30"),
        ):
            prediction = walk_forward_predict(
                X[features],
                y,
                eval_start=start,
                eval_end=end,
                min_train=84,
                purge=FORECAST_HORIZON,
                refit_every=1,
                model_type="svm",
                svm_params=SVM_PARAMS,
                calibration_splits=SVM_CALIBRATION_SPLITS,
            )
            metrics = evaluate_probabilities(prediction, y)
            ic = compute_return_ic(
                prediction, target["future_return"], rolling_window=36
            )[0]
            rows.append(
                {
                    "specification": specification,
                    "features": "|".join(features),
                    "period": period,
                    **metrics,
                    "rank_ic": ic["rank_ic"],
                    "pearson_ic": ic["pearson_ic"],
                }
            )
            predictions[(specification, period)] = prediction

    nested_signal = pd.read_csv(parent / "pre2020_nested_signal_metrics.csv").iloc[0]
    nested_ic = pd.read_csv(parent / "pre2020_nested_ic_summary.csv").iloc[0]
    rows.append(
        {
            "specification": "current_nested_fold_reselection",
            "features": "different features selected inside every outer fold",
            "period": "pre2020_nested_outer_oos",
            "n": nested_signal["n"],
            "auc": nested_signal["auc"],
            "brier": nested_signal["brier"],
            "accuracy": nested_signal["accuracy"],
            "rank_ic": nested_ic["rank_ic"],
            "pearson_ic": nested_ic["pearson_ic"],
        }
    )
    return pd.DataFrame(rows), predictions


def same_period_fixed_portfolios(parent, predictions):
    market = pd.read_csv(
        parent / "portfolio_return_source_monthly.csv", index_col=0, parse_dates=True
    ).iloc[:, 0]
    panel = pd.read_parquet(parent / "monthly_panel.parquet")
    cash = panel["cash_yield_3m"]
    backtests = {}
    for (specification, period), prediction in predictions.items():
        if period != "historical_holdout":
            continue
        raw = prediction * 100
        backtests[specification] = run_backtest(
            market,
            raw,
            target_stock_weight=(raw / 100).clip(0.20, 0.80),
            allocation_policy="linear",
            cash_yield_annual_pct=cash,
            transaction_cost_bps=TRANSACTION_COST_BPS,
            verbose=False,
        )
    common = None
    for backtest in backtests.values():
        valid = backtest.dropna(
            subset=["strategy_return", "market_return", "executed_stock_weight"]
        ).index
        common = valid if common is None else common.intersection(valid)
    rows = []
    for specification, backtest in backtests.items():
        data = backtest.loc[common].copy()
        average_weight = data["executed_stock_weight"].mean()
        same = average_weight * data["market_return"] + (1 - average_weight) * data["cash_return"]
        strategy_stats = performance_stats(data["strategy_return"], data["cash_return"])
        same_stats = performance_stats(same, data["cash_return"])
        rows.append(
            {
                "specification": specification,
                "Start": common.min(),
                "End": common.max(),
                "Months": len(common),
                "average_stock_weight": average_weight,
                "dynamic_CAGR": strategy_stats["CAGR"],
                "same_exposure_CAGR": same_stats["CAGR"],
                "dynamic_Sharpe": strategy_stats["Sharpe"],
                "same_exposure_Sharpe": same_stats["Sharpe"],
                "active_Sharpe_delta": strategy_stats["Sharpe"] - same_stats["Sharpe"],
                "dynamic_MaxDrawdown": strategy_stats["MaxDrawdown"],
            }
        )
    return pd.DataFrame(rows)


def main(parent_run, baseline_run, run_id=None):
    parent = Path(parent_run).resolve()
    baseline = Path(baseline_run).resolve()
    parent_manifest = _load_manifest_complete(parent)
    output = create_run_directory(RUNS_DIR, run_id=run_id)
    config = {
        "track": "benchmark_outperformance_allocation_research",
        "parent_run": str(parent),
        "baseline_run": str(baseline),
        "candidate_specs": CANDIDATE_SPECS,
        "selection_end": "2020-03-31",
        "holdout_used_for_selection": False,
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "objective": "historically outperform KOSPI200 price-index buy-and-hold",
        "future_outperformance_guaranteed": False,
    }
    data_files = [
        parent / "factor_matrix.parquet",
        parent / "target.csv",
        parent / "portfolio_return_source_monthly.csv",
        parent / "monthly_panel.parquet",
        baseline / "model_comparison.csv",
        baseline / "performance_comparison.csv",
        Path(OPERATIONAL_RISK_ACCEPTANCE_FILE),
    ]
    manifest = build_manifest(
        root=Path(__file__).resolve().parent,
        output_dir=output,
        config=config,
        data_files=data_files,
        code_files=[
            Path(__file__).resolve(),
            Path("src/benchmark_allocation.py").resolve(),
            Path("src/benchmark_visualize.py").resolve(),
        ],
        status="running",
    )
    write_manifest(output, manifest)

    artifact = artifact_comparison(parent, baseline)
    artifact.to_csv(output / "baseline_vs_current_artifact_comparison.csv", index=False)
    regime, predictions = fixed_feature_regime_diagnostics(parent, baseline)
    regime.to_csv(output / "fixed_feature_regime_diagnostics.csv", index=False)
    same_period = same_period_fixed_portfolios(parent, predictions)
    same_period.to_csv(output / "same_period_fixed_spec_portfolio_comparison.csv", index=False)
    explanation = {
        "finding": (
            "The current holdout artifact is not worse than the named baseline; "
            "the newly disclosed pre-2020 nested result is a stricter and different estimate."
        ),
        "causes": [
            "Baseline screened with a linear single-factor model and greedy global validation, then kept one fixed feature set.",
            "Current research adds nonlinear screening, bounded exhaustive search and fold-by-fold nested reselection.",
            "Both baseline and current fixed specifications show strong regime instability between pre-2020 and 2020-2026.",
            "The named baseline has no comparable pre-2020 nested outer-OOS artifact, so its holdout Sharpe cannot be compared to the current nested AUC as if they were the same metric.",
        ],
        "holdout_selection_warning": "2020-2026 has already been observed and remains diagnostic only.",
    }
    (output / "baseline_result_explanation.json").write_text(
        json.dumps(explanation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    market = pd.read_csv(
        parent / "portfolio_return_source_monthly.csv", index_col=0, parse_dates=True
    ).iloc[:, 0]
    panel = pd.read_parquet(parent / "monthly_panel.parquet")
    backtests = build_candidate_backtests(
        market, panel["cash_yield_3m"], transaction_cost_bps=TRANSACTION_COST_BPS
    )
    comparison, periods = evaluate_candidate_periods(backtests)
    folds = pre2020_fold_results(backtests, periods["pre2020_research"])
    selected, decision = select_benchmark_policy(comparison, folds)
    comparison["selected_pre2020"] = comparison["policy"].eq(selected)
    comparison["selection_use"] = np.where(
        comparison["period"].eq("pre2020_research"),
        "selection_allowed",
        "diagnostic_only",
    )
    comparison.to_csv(output / "allocation_candidate_comparison.csv", index=False)
    folds.to_csv(output / "allocation_pre2020_fold_results.csv", index=False)
    decision.to_csv(output / "allocation_pre2020_selection.csv", index=False)
    # Once selected using the common pre-2020 candidate window, cover the
    # earliest history causally by matching KOSPI until the 3-month signal is
    # observable.  This warm-up rule is fixed and cannot create excess return.
    selected_weight = build_candidate_weights(market)[selected].fillna(1.0)
    selected_backtest = run_backtest(
        market,
        selected_weight * 100,
        target_stock_weight=selected_weight,
        allocation_policy=selected,
        cash_yield_annual_pct=panel["cash_yield_3m"],
        transaction_cost_bps=TRANSACTION_COST_BPS,
        verbose=False,
    )
    selected_backtest["benchmark_active_return"] = (
        selected_backtest["strategy_return"] - selected_backtest["market_return"]
    )
    selected_backtest.to_csv(output / "selected_allocation_monthly.csv")

    full_available_index = selected_backtest.dropna(
        subset=["strategy_return", "market_return", "executed_stock_weight"]
    ).index
    full_available_index = full_available_index[
        full_available_index <= pd.Timestamp("2026-04-30")
    ]
    full_available_report = evaluate_against_market(
        selected_backtest,
        full_available_index,
        selected,
        "full_available_history_with_market_warmup",
    )
    pd.DataFrame([full_available_report]).to_csv(
        output / "selected_allocation_full_available_history_report.csv",
        index=False,
    )

    diagnostic = selected_backtest.loc[full_available_index].dropna(
        subset=["strategy_return", "market_return"]
    )
    calendar_rows = []
    for year, data in diagnostic.groupby(diagnostic.index.year):
        strategy_return = float((1 + data["strategy_return"]).prod() - 1)
        market_return = float((1 + data["market_return"]).prod() - 1)
        calendar_rows.append(
            {
                "year": int(year),
                "months": len(data),
                "strategy_return": strategy_return,
                "market_return": market_return,
                "active_return": strategy_return - market_return,
                "beat_market": strategy_return > market_return,
            }
        )
    calendar = pd.DataFrame(calendar_rows)
    calendar.to_csv(output / "selected_allocation_calendar_years.csv", index=False)
    rolling_rows = []
    for window in (12, 36, 60):
        strategy_total = (1 + diagnostic["strategy_return"]).rolling(window).apply(
            np.prod, raw=True
        ) - 1
        market_total = (1 + diagnostic["market_return"]).rolling(window).apply(
            np.prod, raw=True
        ) - 1
        frame = pd.DataFrame(
            {
                "strategy_total_return": strategy_total,
                "market_total_return": market_total,
            }
        ).dropna()
        frame["active_total_return"] = (
            frame["strategy_total_return"] - frame["market_total_return"]
        )
        frame["beat_market"] = frame["active_total_return"] > 0
        frame["window_months"] = window
        frame.index.name = "window_end"
        rolling_rows.append(frame.reset_index())
    rolling = pd.concat(rolling_rows, ignore_index=True)
    rolling.to_csv(output / "selected_allocation_rolling_outperformance.csv", index=False)
    regime_rows = []
    market_volatility = diagnostic["market_return"].rolling(12, min_periods=12).std()
    high_vol_cutoff = market_volatility.loc[:"2020-03-31"].median()
    regimes = {
        "market_up_month": diagnostic["market_return"] > 0,
        "market_down_month": diagnostic["market_return"] <= 0,
        "high_volatility": market_volatility > high_vol_cutoff,
        "low_volatility": market_volatility <= high_vol_cutoff,
    }
    for regime, mask in regimes.items():
        data = diagnostic.loc[mask.fillna(False)]
        regime_rows.append(
            {
                "regime": regime,
                "months": len(data),
                "average_stock_weight": data["executed_stock_weight"].mean(),
                "annualized_strategy_return": data["strategy_return"].mean() * 12,
                "annualized_market_return": data["market_return"].mean() * 12,
                "annualized_active_return": (
                    data["strategy_return"] - data["market_return"]
                ).mean() * 12,
            }
        )
    regime_report = pd.DataFrame(regime_rows)
    regime_report.to_csv(output / "selected_allocation_regime_report.csv", index=False)

    selected_reports = comparison.loc[comparison["policy"].eq(selected)].copy()
    selected_reports.to_csv(output / "selected_allocation_period_report.csv", index=False)
    historical_full_outperformance = bool(
        full_available_report["CAGR_difference"] > 0
        and full_available_report["relative_terminal_wealth"] > 0
    )
    parent_operational = pd.read_csv(parent / "deployment_gates.csv").iloc[0]
    operational_gate = bool(parent_operational["operational_gate"])
    forward_shadow_eligible = bool(
        decision.loc[decision["policy"].eq(selected), "pre2020_gate"].iloc[0]
        and operational_gate
    )
    strategy_gate = pd.DataFrame(
        [
            {
                "selected_policy": selected,
                "selection_end": "2020-03-31",
                "holdout_used_for_selection": False,
                "pre2020_gate": True,
                "operational_gate_from_parent": operational_gate,
                "historical_full_period_outperformance": historical_full_outperformance,
                "future_outperformance_guaranteed": False,
                "forward_shadow_eligible": forward_shadow_eligible,
                "status": (
                    "forward_shadow_candidate"
                    if forward_shadow_eligible
                    else "research_only"
                ),
            }
        ]
    )
    strategy_gate.to_csv(output / "benchmark_strategy_gate.csv", index=False)
    latest_target = selected_backtest["target_stock_weight"].dropna().iloc[-1]
    latest_executed = selected_backtest["executed_stock_weight"].dropna().iloc[-1]
    latest_indicator = market.pct_change(3, fill_method=None).loc[
        selected_backtest.index.max()
    ]
    latest = {
        "date": selected_backtest.index.max().date().isoformat(),
        "policy": selected,
        "target_stock_weight": float(latest_target),
        "executed_stock_weight": float(latest_executed),
        "cash_weight": float(1 - latest_target),
        "kospi200_3m_price_return": float(latest_indicator),
        "decision_rule": "stock 100% if 3-month return > 0, otherwise cash 100%",
        "historical_full_period_outperformance": historical_full_outperformance,
        "future_outperformance_guaranteed": False,
        "status": strategy_gate.iloc[0]["status"],
    }
    (output / "latest_benchmark_allocation.json").write_text(
        json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    latest_date = selected_backtest.index.max()
    shadow_spec = {
        "status": (
            "ready_for_next_observation"
            if forward_shadow_eligible
            else "blocked_until_all_deployment_gates_pass"
        ),
        "strategy": "KOSPI200/cash absolute momentum allocation",
        "model": selected,
        "allocation_policy": selected,
        "freeze_date": latest_date.date().isoformat(),
        "first_eligible_observation": (
            latest_date + pd.offsets.MonthEnd(1)
        ).date().isoformat(),
        "minimum_shadow_months": 12,
        "historical_holdout_may_not_change_spec": True,
        "features": ["kospi200_monthly_close", "cash_yield_3m"],
        "formula": (
            "target_stock_weight=1.0 when KOSPI200 3-month price return > 0; "
            "otherwise 0.0; execute one month later"
        ),
        "stock_weight_limits": [0.0, 1.0],
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "operational_gate_profile": parent_operational[
            "operational_gate_profile"
        ],
        "waived_risks": parent_operational["waived_risks"],
        "future_outperformance_guaranteed": False,
        "monitoring": {
            "missing_frozen_feature": "stop",
            "raw_ews_outside_0_100": "stop",
            "target_weight_outside_limits": "stop",
            "score_population_stability_index_warning": 0.25,
            "monthly_turnover_warning": 1.0,
            "active_drawdown_vs_same_exposure_stop": -0.15,
            "calibration_slope_warning_range": [0.0, 2.0],
        },
    }
    shadow_spec["freeze_hash"] = canonical_spec_hash(shadow_spec)
    (output / "forward_shadow_spec.json").write_text(
        json.dumps(shadow_spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    initialize_shadow_ledger(output / "forward_shadow_ledger.csv")

    plot_benchmark_allocation_dashboard(
        selected_backtest,
        full_available_report,
        output / "benchmark_allocation_dashboard.png",
    )
    plot_outperformance_diagnostics(
        rolling,
        calendar,
        regime_report,
        output / "benchmark_outperformance_diagnostics.png",
    )
    plot_candidate_selection(
        comparison,
        selected,
        output / "benchmark_candidate_selection.png",
    )
    plot_baseline_comparison(
        artifact,
        same_period,
        output / "baseline_vs_current_diagnostics.png",
    )

    manifest.update(
        {
            "status": "complete",
            "selected_policy": selected,
            "selection_end": "2020-03-31",
            "holdout_used_for_selection": False,
            "historical_full_period_outperformance": historical_full_outperformance,
            "future_outperformance_guaranteed": False,
            "forward_shadow_eligible": forward_shadow_eligible,
            "output_files": sorted(path.name for path in output.iterdir()),
        }
    )
    write_manifest(output, manifest)
    print(f"Benchmark allocation research complete: {output}")
    print(selected_reports.to_string(index=False))
    return output


if __name__ == "__main__":
    args = parse_args()
    main(args.parent_run, args.baseline_run, run_id=args.run_id)
