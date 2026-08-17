"""Run the continuous 3-month-return experiment separately from classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics import performance_stats
from src.backtest import run_backtest
from src.config import (
    FORECAST_HORIZON,
    PERCENTILE_BREAKS,
    PERCENTILE_MIN_HISTORY,
    PERCENTILE_WEIGHTS,
    RANDOM_SEED,
    RESEARCH_HOLDOUT_START,
    RUNS_DIR,
    TRANSACTION_COST_BPS,
)
from src.experiment import build_manifest, create_run_directory, write_manifest
from src.position_sizing import expanding_percentile_weight
from src.regression import (
    REGRESSION_MODELS,
    regression_metrics,
    select_regression_model,
    walk_forward_regression,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification-run", required=True)
    parser.add_argument("--run-id")
    return parser.parse_args()


def portfolio_metrics(
    prediction,
    market,
    cash_yield,
    *,
    evaluation_start=None,
    evaluation_end=None,
):
    target_weight = expanding_percentile_weight(
        prediction,
        breaks=PERCENTILE_BREAKS,
        weights=PERCENTILE_WEIGHTS,
        min_history=PERCENTILE_MIN_HISTORY,
    )
    backtest = run_backtest(
        market_price=market,
        ews=prediction,
        target_stock_weight=target_weight,
        allocation_policy="regression_expanding_percentile",
        cash_yield_annual_pct=cash_yield,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        verbose=False,
    )
    valid = backtest.dropna(
        subset=["strategy_return", "executed_stock_weight", "cash_return"]
    ).copy()
    if evaluation_start is not None:
        valid = valid.loc[valid.index >= pd.Timestamp(evaluation_start)]
    if evaluation_end is not None:
        valid = valid.loc[valid.index <= pd.Timestamp(evaluation_end)]
    if len(valid) < 12:
        return {}, backtest
    average_weight = float(valid["executed_stock_weight"].mean())
    same_return = (
        average_weight * valid["market_return"]
        + (1 - average_weight) * valid["cash_return"]
    )
    strategy_stats = performance_stats(valid["strategy_return"], valid["cash_return"])
    same_stats = performance_stats(same_return, valid["cash_return"])
    active = valid["strategy_return"] - same_return
    return {
        "portfolio_months": len(valid),
        "average_stock_weight": average_weight,
        "strategy_sharpe": strategy_stats.get("Sharpe", np.nan),
        "same_exposure_sharpe": same_stats.get("Sharpe", np.nan),
        "active_sharpe_delta": (
            strategy_stats.get("Sharpe", np.nan) - same_stats.get("Sharpe", np.nan)
        ),
        "annualized_active_return": float(active.mean() * 12),
    }, backtest


def main(classification_run, run_id=None):
    source_run = Path(classification_run).resolve()
    required = [
        "experiment_manifest.json",
        "factor_matrix.parquet",
        "target.csv",
        "outer_validation_folds.csv",
        "outer_fold_feature_selections.csv",
        "selected_features.json",
        "monthly_panel.parquet",
        "portfolio_return_source_monthly.csv",
    ]
    missing = [name for name in required if not (source_run / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Classification run is incomplete: {missing}")
    source_manifest = json.loads(
        (source_run / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    if source_manifest.get("status") != "complete":
        raise ValueError("Regression requires an immutable complete classification run")

    output = create_run_directory(RUNS_DIR, run_id=run_id)
    config = {
        "track": "regression_research_separate",
        "parent_classification_run": str(source_run),
        "target": "forward_3m_kospi200_price_return",
        "models": list(REGRESSION_MODELS),
        "target_preprocessing": "train-only 1/99% winsorization and standardization",
        "selection_period_end": "2020-03-31",
        "research_holdout_start": RESEARCH_HOLDOUT_START,
        "model_selection_rule": "positive pre-holdout outer-OOS Rank IC, RMSE tie-break",
        "holdout_used_for_selection": False,
        "multiple_comparison_note": "three pre-declared regression algorithms",
        "random_seed": RANDOM_SEED,
    }
    manifest = build_manifest(
        root=Path(__file__).resolve().parent,
        output_dir=output,
        config=config,
        data_files=[source_run / name for name in required],
        code_files=[Path(__file__).resolve(), Path("src/regression.py").resolve()],
        status="running",
    )
    write_manifest(output, manifest)

    X = pd.read_parquet(source_run / "factor_matrix.parquet")
    target_frame = pd.read_csv(
        source_run / "target.csv", index_col=0, parse_dates=True
    )
    future_return = target_frame["future_return"].reindex(X.index)
    folds = pd.read_csv(source_run / "outer_validation_folds.csv", parse_dates=[
        "development_end", "inner_validation_start", "inner_validation_end",
        "outer_start", "outer_end",
    ])
    selections = pd.read_csv(source_run / "outer_fold_feature_selections.csv")

    predictions = {model: [] for model in REGRESSION_MODELS}
    audit_parts = []
    for _, fold in folds.iterrows():
        features = selections.loc[
            selections["fold"].eq(fold["fold"]), "feature"
        ].tolist()
        if not features:
            raise ValueError(f"No frozen features for outer fold {fold['fold']}")
        for model in REGRESSION_MODELS:
            prediction, audit = walk_forward_regression(
                X[features], future_return,
                eval_start=fold["outer_start"], eval_end=fold["outer_end"],
                model_type=model, min_train=84, purge=FORECAST_HORIZON,
                random_seed=RANDOM_SEED,
            )
            predictions[model].append(prediction)
            audit_parts.append(audit.assign(
                fold=int(fold["fold"]), model=model, features="|".join(features)
            ))

    outer_predictions = pd.concat(
        {model: pd.concat(parts).sort_index() for model, parts in predictions.items()},
        axis=1,
    )
    outer_predictions.to_csv(output / "pre2020_outer_predictions.csv")
    pd.concat(audit_parts, ignore_index=True).to_csv(
        output / "train_only_preprocessing_audit.csv", index=False
    )

    market_frame = pd.read_csv(
        source_run / "portfolio_return_source_monthly.csv",
        index_col=0,
        parse_dates=True,
    )
    if market_frame.shape[1] != 1:
        raise ValueError("Parent portfolio-return source must contain one series")
    market = market_frame.iloc[:, 0].sort_index()
    panel = pd.read_parquet(source_run / "monthly_panel.parquet")
    cash_yield = panel["cash_yield_3m"]
    metric_rows = []
    pre_backtests = {}
    for model in REGRESSION_MODELS:
        row = {"model": model, **regression_metrics(outer_predictions[model], future_return)}
        portfolio, backtest = portfolio_metrics(outer_predictions[model], market, cash_yield)
        row.update(portfolio)
        metric_rows.append(row)
        pre_backtests[model] = backtest
    pre_metrics = pd.DataFrame(metric_rows)
    selected_model, selection_reason = select_regression_model(pre_metrics)
    pre_metrics["selected_pre2020"] = pre_metrics["model"].eq(selected_model)
    pre_metrics["selection_reason"] = np.where(
        pre_metrics["selected_pre2020"], selection_reason, "not selected"
    )
    pre_metrics.to_csv(output / "pre2020_model_comparison.csv", index=False)
    pre_backtests[selected_model].to_csv(output / "pre2020_selected_backtest.csv")

    selected_features = json.loads(
        (source_run / "selected_features.json").read_text(encoding="utf-8")
    )
    holdout_end = future_return.dropna().index.max()
    holdout_prediction, holdout_audit = walk_forward_regression(
        X[selected_features], future_return,
        eval_start=RESEARCH_HOLDOUT_START, eval_end=holdout_end,
        model_type=selected_model, min_train=84, purge=FORECAST_HORIZON,
        random_seed=RANDOM_SEED,
    )
    holdout_prediction.to_csv(output / "research_holdout_prediction.csv")
    holdout_audit.assign(model=selected_model).to_csv(
        output / "research_holdout_preprocessing_audit.csv", index=False
    )
    holdout_sizing_history = pd.concat(
        [outer_predictions[selected_model], holdout_prediction]
    ).sort_index()
    holdout_sizing_history = holdout_sizing_history[
        ~holdout_sizing_history.index.duplicated(keep="last")
    ]
    holdout_portfolio, holdout_backtest = portfolio_metrics(
        holdout_sizing_history,
        market,
        cash_yield,
        evaluation_start=RESEARCH_HOLDOUT_START,
        evaluation_end=holdout_end,
    )
    holdout_report = pd.DataFrame([{
        "model": selected_model,
        **regression_metrics(holdout_prediction, future_return),
        **holdout_portfolio,
        "selection_use": "diagnostic_only",
        "used_for_model_or_policy_selection": False,
    }])
    holdout_report.to_csv(output / "research_holdout_report.csv", index=False)
    holdout_backtest.to_csv(output / "research_holdout_backtest.csv")

    manifest.update({
        "status": "complete",
        "selected_model": selected_model,
        "selection_reason": selection_reason,
        "selected_features_from_parent": selected_features,
        "holdout_used_for_selection": False,
        "output_files": sorted(path.name for path in output.iterdir()),
    })
    write_manifest(output, manifest)
    print(f"Regression research complete: {output}")
    print(pre_metrics.to_string(index=False))
    return output


if __name__ == "__main__":
    args = parse_args()
    main(args.classification_run, run_id=args.run_id)
