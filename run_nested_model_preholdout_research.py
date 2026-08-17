"""Run an independent PIT-safe nested pre-holdout gate for one model family.

Unlike the main pipeline's historical challenger comparison, this utility
gives Logistic, SVM or MLP its own fold-local screen, combination search,
outer prediction and allocation-policy gate.  The source matrix is truncated
before the configured historical holdout before any model is fit.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from run_pipeline import _fold_eligibility_map, _fold_signal_gate
from src.analytics import compute_return_ic
from src.config import (
    COMBINATION_SELECTION_REFIT_EVERY,
    CORRELATION_THRESHOLD,
    EXHAUSTIVE_COMBO_CANDIDATE_POOL,
    FIXED_BIN_THRESHOLDS,
    FIXED_BIN_WEIGHTS,
    FORECAST_HORIZON,
    GROUP_CANDIDATES_PER_GROUP,
    MAX_FEATURES_PER_BASE,
    MAX_FEATURES_PER_GROUP,
    MAX_MODEL_FEATURES,
    MAX_STOCK_WEIGHT,
    MIN_DISTINCT_GROUPS,
    MIN_MODEL_FEATURES,
    MIN_OOS_PREDICTIONS,
    MIN_STOCK_WEIGHT,
    PERCENTILE_BREAKS,
    PERCENTILE_MIN_HISTORY,
    PERCENTILE_WEIGHTS,
    POSITION_SIZING_POLICIES,
    RANDOM_SEED,
    RAW_TOP_FEATURES_PER_BASE,
    SINGLE_FACTOR_MODEL_TYPE,
    SINGLE_FACTOR_REFIT_EVERY,
    SMOOTHED_LINEAR_SPAN,
    STATIC_FALLBACK_WEIGHT,
    TOP_FEATURE_POOL,
    TRANSACTION_COST_SCENARIOS_BPS,
)
from src.modeling import evaluate_probabilities
from src.validation import (
    compare_position_sizing,
    evaluate_signal_gate,
    nested_outer_predict,
    select_position_policy,
)


MODEL_SPECS = {
    "logistic": {
        "selection_model_type": "logistic",
        "final_model_type": "logistic",
        "svm_params": {},
        "mlp_params": {},
    },
    "svm": {
        "selection_model_type": "svm_rank",
        "final_model_type": "svm",
        "svm_params": {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
        "mlp_params": {},
    },
    "mlp": {
        "selection_model_type": "logistic",
        "final_model_type": "mlp",
        "svm_params": {},
        "mlp_params": {
            "hidden_layer_sizes": (4,),
            "activation": "tanh",
            "solver": "lbfgs",
            "alpha": 0.10,
            "max_iter": 1000,
            "tol": 1e-4,
            "shuffle": False,
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    return parser.parse_args()


def _sizing_config():
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


def _load_folds(run_dir: Path):
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


def run_research(run_dir: Path, output_dir: Path, *, model_name: str):
    spec = MODEL_SPECS[model_name]
    manifest = json.loads(
        (run_dir / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    holdout_start = pd.Timestamp(manifest["config"]["research_holdout_start"])
    research_end = holdout_start - pd.offsets.MonthEnd(1)
    X = pd.read_parquet(run_dir / "factor_matrix.parquet").loc[:research_end]
    target = pd.read_csv(run_dir / "target.csv", index_col=0, parse_dates=True)
    target = target.loc[:research_end]
    # Labels at the end of the development sample must not use returns from
    # the historical holdout.  Keep the feature dates for prediction and
    # portfolio scoring, but mask labels whose full forecast path is not yet
    # observable before the declared cutoff.
    label_end = (
        holdout_start.to_period("M") - 1 - FORECAST_HORIZON
    ).to_timestamp("M")
    target.loc[target.index > label_end, ["future_return", "y"]] = np.nan
    candidates = pd.read_csv(
        run_dir / "mlp_deployment_candidate_universe.csv"
    )["feature"].tolist()
    metadata = pd.read_csv(run_dir / "factor_candidates.csv")
    feature_groups = (
        metadata.drop_duplicates("base").set_index("base")["group"].to_dict()
    )
    folds = _load_folds(run_dir)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    prediction, selections, screening = nested_outer_predict(
        X=X[candidates],
        y=target["y"],
        folds=folds,
        feature_groups=feature_groups,
        screening_model_type=SINGLE_FACTOR_MODEL_TYPE,
        selection_model_type=spec["selection_model_type"],
        final_model_type=spec["final_model_type"],
        min_train_months=84,
        final_min_train_months=84,
        horizon=FORECAST_HORIZON,
        single_factor_refit_every=SINGLE_FACTOR_REFIT_EVERY,
        min_oos_predictions=MIN_OOS_PREDICTIONS,
        top_feature_pool=TOP_FEATURE_POOL,
        raw_top_features_per_base=RAW_TOP_FEATURES_PER_BASE,
        group_candidates_per_group=GROUP_CANDIDATES_PER_GROUP,
        correlation_threshold=CORRELATION_THRESHOLD,
        combination_candidate_pool=20,
        exhaustive_candidate_pool=min(6, EXHAUSTIVE_COMBO_CANDIDATE_POOL),
        min_model_features=MIN_MODEL_FEATURES,
        max_model_features=min(6, MAX_MODEL_FEATURES),
        min_validation_improvement=0.0,
        max_features_per_base=MAX_FEATURES_PER_BASE,
        max_features_per_group=MAX_FEATURES_PER_GROUP,
        min_distinct_groups=MIN_DISTINCT_GROUPS,
        svm_params=spec["svm_params"],
        mlp_params=spec["mlp_params"],
        calibration_splits=3,
        random_state=RANDOM_SEED,
        combination_refit_every=COMBINATION_SELECTION_REFIT_EVERY,
        n_jobs=-1,
        final_refit_every=1,
        allow_unavailable_folds=True,
    )
    completed = set(selections["fold"])
    gate_folds = [fold for fold in folds if fold.fold in completed]
    metrics = evaluate_probabilities(prediction, target["y"])
    ic, _, _ = compute_return_ic(prediction, target["future_return"], rolling_window=36)
    fold_rows = _fold_signal_gate(
        gate_folds,
        prediction,
        target["y"],
        target["future_return"],
        eligibility_starts=_fold_eligibility_map(selections),
    )
    signal_summary, fold_details = evaluate_signal_gate(
        fold_rows,
        aggregate_auc=metrics["auc"],
        aggregate_rank_ic=ic["rank_ic"],
    )
    panel = pd.read_parquet(run_dir / "monthly_panel.parquet").loc[:research_end]
    price = pd.read_csv(
        run_dir / "portfolio_return_source_monthly.csv",
        index_col=0,
        parse_dates=True,
    ).iloc[:, 0].loc[:research_end]
    labels = pd.Series(np.nan, index=prediction.index)
    for fold in gate_folds:
        labels.loc[fold.outer_start : fold.outer_end] = fold.fold
    comparison, monthly, fold_results = compare_position_sizing(
        market_price=price,
        raw_ews=(prediction * 100).rename("raw_ews"),
        cash_yield=panel["cash_yield_3m"],
        policies=POSITION_SIZING_POLICIES,
        transaction_cost_scenarios=TRANSACTION_COST_SCENARIOS_BPS,
        sizing_config=_sizing_config(),
        fold_labels=labels,
        evaluation_end=research_end,
    )
    policy, decisions = select_position_policy(
        comparison, fold_results, baseline="static_50_50"
    )
    decision = decisions.loc[decisions["policy"].eq(policy)].iloc[0]
    last_fold = max(completed)
    locked_features = selections.loc[
        selections["fold"].eq(last_fold)
    ].sort_values("selection_rank")["feature"].tolist()
    summary = {
        "model": model_name,
        "candidate_universe": "strict_pit_safe_compact",
        "completed_folds": len(gate_folds),
        "aggregate_auc": signal_summary["aggregate_auc"],
        "aggregate_rank_ic": signal_summary["aggregate_rank_ic"],
        "fold_joint_direction_pass_ratio": signal_summary[
            "fold_joint_direction_pass_ratio"
        ],
        "signal_gate_passed": bool(signal_summary["signal_gate_passed"]),
        "selected_policy": policy,
        "portfolio_gate_passed": bool(decision["portfolio_gate_passed"]),
        "median_fold_sharpe_difference": decision[
            "median_fold_Sharpe_difference"
        ],
        "positive_fold_ratio": decision["positive_fold_ratio"],
        "annualized_active_return_10bps": decision[
            "annualized_active_return_10bps"
        ],
        "locked_feature_source_fold": last_fold,
        "locked_features": "|".join(locked_features),
        "historical_holdout_opened": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)
    prediction.to_csv(output_dir / "prediction.csv")
    selections.to_csv(output_dir / "selections.csv", index=False)
    screening.to_csv(output_dir / "screening.csv", index=False)
    fold_details.to_csv(output_dir / "fold_signal.csv", index=False)
    comparison.to_csv(output_dir / "policy_comparison.csv", index=False)
    decisions.to_csv(output_dir / "policy_gate.csv", index=False)
    monthly.to_csv(output_dir / "policy_monthly.csv")
    fold_results.to_csv(output_dir / "policy_folds.csv", index=False)
    (output_dir / "locked_features.json").write_text(
        json.dumps(locked_features, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame([summary]).to_string(index=False))
    return summary


def main():
    args = parse_args()
    run_research(
        Path(args.run_dir).resolve(),
        Path(args.output_dir).resolve(),
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
