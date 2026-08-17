"""Evaluate one cross-market linear-anchored MLP protocol before 2020.

The shrinkage path is deliberately small and identical for KOSPI200, S&P 500
and NASDAQ-100.  Each market selects the *least* linear anchoring that clears
both gates; it never maximizes its own historical return.  Every input matrix,
target and return series is truncated at the declared research holdout boundary
before prediction or policy selection.  Anchor weight 1.0 is reported only as
an endpoint diagnostic and is excluded from deployable selection, so an MLP
remains part of every candidate.  Weight 0.0 is the pure-MLP baseline used to
avoid adding an anchor where the nonlinear model already works.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from run_pipeline import _fold_eligibility_map, _fold_signal_gate
from src.analytics import compute_return_ic
from src.config import (
    FIXED_BIN_THRESHOLDS,
    FIXED_BIN_WEIGHTS,
    FORECAST_HORIZON,
    MAX_STOCK_WEIGHT,
    MIN_STOCK_WEIGHT,
    PERCENTILE_BREAKS,
    PERCENTILE_MIN_HISTORY,
    PERCENTILE_WEIGHTS,
    POSITION_SIZING_POLICIES,
    RANDOM_SEED,
    SMOOTHED_LINEAR_SPAN,
    STATIC_FALLBACK_WEIGHT,
    TRANSACTION_COST_SCENARIOS_BPS,
)
from src.modeling import evaluate_probabilities
from src.validation import (
    compare_position_sizing,
    evaluate_signal_gate,
    fixed_outer_predict,
    select_position_policy,
)


# The final two values are diagnostics around the linear endpoint.  A 1.00
# anchor is deliberately non-deployable (it contains no nonlinear component),
# but it shows whether the MLP residual or the underlying feature set is the
# source of a failed gate.
ANCHOR_WEIGHTS = (0.00, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def sizing_config():
    return {
        "min_weight": MIN_STOCK_WEIGHT,
        "max_weight": MAX_STOCK_WEIGHT,
        "fixed_thresholds": FIXED_BIN_THRESHOLDS,
        "fixed_weights": FIXED_BIN_WEIGHTS,
        "percentile_breaks": PERCENTILE_BREAKS,
        "percentile_weights": PERCENTILE_WEIGHTS,
        "percentile_min_history": PERCENTILE_MIN_HISTORY,
        "smoothing_span": SMOOTHED_LINEAR_SPAN,
        "static_stock_weight": STATIC_FALLBACK_WEIGHT,
    }


def load_folds(run_dir: Path):
    frame = pd.read_csv(
        run_dir / "mlp_outer_validation_folds.csv",
        parse_dates=[
            "development_end",
            "inner_validation_start",
            "inner_validation_end",
            "outer_start",
            "outer_end",
        ],
    )
    return [SimpleNamespace(**row._asdict()) for row in frame.itertuples(index=False)]


def load_market_inputs(run_dir: Path):
    manifest = json.loads(
        (run_dir / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    config = manifest["config"]
    market_key = config["market_key"]
    holdout_start = pd.Timestamp(config["research_holdout_start"])
    research_end = holdout_start - pd.offsets.MonthEnd(1)
    label_end = (
        holdout_start.to_period("M") - 1 - FORECAST_HORIZON
    ).to_timestamp("M")
    X = pd.read_parquet(run_dir / "factor_matrix.parquet").loc[:research_end]
    target_file = "target.csv" if market_key == "kospi200" else "mlp_target.csv"
    target = pd.read_csv(
        run_dir / target_file, index_col=0, parse_dates=True
    ).loc[:research_end]
    target.loc[target.index > label_end, "y"] = np.nan
    target.loc[target.index > label_end, "future_return"] = np.nan
    features = list(manifest["model_feature_sets"]["mlp"])
    spec = json.loads(
        (run_dir / "mlp_research_shadow_spec.json").read_text(encoding="utf-8")
    )
    price = pd.read_csv(
        run_dir / "portfolio_return_source_monthly.csv",
        index_col=0,
        parse_dates=True,
    ).iloc[:, 0].loc[:research_end]
    panel = pd.read_parquet(run_dir / "monthly_panel.parquet").loc[:research_end]
    return {
        "market_key": market_key,
        "market_name": config["market_name"],
        "research_end": research_end,
        "X": X,
        "target": target,
        "features": features,
        "mlp_params": spec["model_params"],
        "folds": load_folds(run_dir),
        "price": price,
        "cash_yield": panel["cash_yield_3m"],
    }


def fixed_predictions(inputs, model_type):
    prediction, selections, _ = fixed_outer_predict(
        X=inputs["X"],
        y=inputs["target"]["y"],
        folds=inputs["folds"],
        features=inputs["features"],
        feature_groups={},
        final_model_type=model_type,
        final_min_train_months=84,
        horizon=FORECAST_HORIZON,
        refit_every=1,
        mlp_params=inputs["mlp_params"] if model_type == "mlp" else None,
        random_state=RANDOM_SEED,
        selection_note="cross_market_anchor_research;holdout_not_opened",
    )
    completed = set(selections["fold"])
    folds = [fold for fold in inputs["folds"] if fold.fold in completed]
    return prediction, selections, folds


def evaluate_candidate(inputs, prediction, selections, folds, anchor_weight):
    metrics = evaluate_probabilities(prediction, inputs["target"]["y"])
    ic, _, _ = compute_return_ic(
        prediction, inputs["target"]["future_return"], rolling_window=36
    )
    fold_rows = _fold_signal_gate(
        folds,
        prediction,
        inputs["target"]["y"],
        inputs["target"]["future_return"],
        eligibility_starts=_fold_eligibility_map(selections),
    )
    signal, _ = evaluate_signal_gate(
        fold_rows,
        aggregate_auc=metrics["auc"],
        aggregate_rank_ic=ic["rank_ic"],
    )
    labels = pd.Series(np.nan, index=prediction.index)
    for fold in folds:
        labels.loc[fold.outer_start : fold.outer_end] = fold.fold
    comparison, monthly, fold_results = compare_position_sizing(
        market_price=inputs["price"],
        raw_ews=(prediction * 100).rename("raw_ews"),
        cash_yield=inputs["cash_yield"],
        policies=POSITION_SIZING_POLICIES,
        transaction_cost_scenarios=TRANSACTION_COST_SCENARIOS_BPS,
        sizing_config=sizing_config(),
        fold_labels=labels,
        evaluation_end=inputs["research_end"],
    )
    policy, decisions = select_position_policy(
        comparison, fold_results, baseline="static_50_50"
    )
    decision = decisions.loc[decisions["policy"].eq(policy)].iloc[0]
    row = {
        "market_key": inputs["market_key"],
        "market_name": inputs["market_name"],
        "anchor_weight": anchor_weight,
        "mlp_weight": 1.0 - anchor_weight,
        "aggregate_auc": signal["aggregate_auc"],
        "aggregate_rank_ic": signal["aggregate_rank_ic"],
        "fold_joint_direction_pass_ratio": signal[
            "fold_joint_direction_pass_ratio"
        ],
        "signal_gate": bool(signal["signal_gate_passed"]),
        "selected_policy": policy,
        "median_fold_sharpe_difference": decision[
            "median_fold_Sharpe_difference"
        ],
        "positive_fold_ratio": decision["positive_fold_ratio"],
        "annualized_active_return_25bps": decision[
            "annualized_active_return_25bps"
        ],
        "drawdown_difference_25bps": decision["drawdown_difference_25bps"],
        "portfolio_gate": bool(decision["portfolio_gate_passed"]),
        "joint_gate": bool(
            signal["signal_gate_passed"]
            and decision["portfolio_gate_passed"]
            and policy != "static_50_50"
        ),
        "historical_holdout_opened": False,
    }
    return row, comparison, monthly, fold_results, decisions, fold_rows


def run(run_dirs, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for run_dir in map(Path, run_dirs):
        inputs = load_market_inputs(run_dir.resolve())
        print(f"[ANCHOR PRE-2020] {inputs['market_name']}", flush=True)
        mlp_prediction, mlp_selections, mlp_folds = fixed_predictions(inputs, "mlp")
        logistic_prediction, _, _ = fixed_predictions(inputs, "logistic")
        common = mlp_prediction.dropna().index.intersection(
            logistic_prediction.dropna().index
        )
        for anchor_weight in ANCHOR_WEIGHTS:
            prediction = pd.Series(np.nan, index=mlp_prediction.index, dtype=float)
            prediction.loc[common] = (
                anchor_weight * logistic_prediction.loc[common]
                + (1.0 - anchor_weight) * mlp_prediction.loc[common]
            )
            result = evaluate_candidate(
                inputs,
                prediction,
                mlp_selections,
                mlp_folds,
                anchor_weight,
            )
            row, comparison, monthly, fold_results, decisions, fold_rows = result
            all_rows.append(row)
            stem = output_dir / (
                f"{inputs['market_key']}_anchor_{int(anchor_weight * 100):02d}"
            )
            prediction.to_csv(f"{stem}_prediction.csv")
            comparison.to_csv(f"{stem}_policy_comparison.csv", index=False)
            monthly.to_csv(f"{stem}_policy_monthly.csv")
            fold_results.to_csv(f"{stem}_policy_folds.csv", index=False)
            decisions.to_csv(f"{stem}_policy_gate.csv", index=False)
            fold_rows.to_csv(f"{stem}_signal_folds.csv", index=False)
    summary = pd.DataFrame(all_rows)
    weight_summary = (
        summary.groupby("anchor_weight", as_index=False)
        .agg(
            markets=("market_key", "nunique"),
            markets_joint_pass=("joint_gate", "sum"),
            minimum_direction=("fold_joint_direction_pass_ratio", "min"),
            minimum_fold_sharpe=("median_fold_sharpe_difference", "min"),
            minimum_active_return_25bps=("annualized_active_return_25bps", "min"),
        )
    )
    selected_by_market = {}
    for market_key, market_rows in summary.groupby("market_key"):
        eligible = market_rows.loc[
            market_rows["joint_gate"] & market_rows["anchor_weight"].lt(1.0)
        ]
        selected_by_market[market_key] = (
            float(eligible.sort_values("anchor_weight").iloc[0]["anchor_weight"])
            if not eligible.empty
            else None
        )
    summary["selected_market_weight"] = summary.apply(
        lambda row: row["anchor_weight"] == selected_by_market[row["market_key"]],
        axis=1,
    )
    summary.to_csv(output_dir / "cross_market_candidate_summary.csv", index=False)
    weight_summary.to_csv(output_dir / "universal_weight_selection.csv", index=False)
    (output_dir / "selection.json").write_text(
        json.dumps(
            {
                "selected_anchor_weight_by_market": selected_by_market,
                "candidate_weights": list(ANCHOR_WEIGHTS),
                "selection_scope": "pre2020_all_markets_only",
                "selection_rule": (
                    "per_market_least_anchor_weight_that_clears_both_gates"
                ),
                "historical_holdout_opened": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(weight_summary.to_string(index=False))
    return selected_by_market


def main():
    args = parse_args()
    run(args.run_dir, Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
