"""Pre-holdout-only nested MLP stability research for one completed run.

The script never reads observations on or after the configured research
holdout.  Results are architecture-development evidence only; the chosen
procedure must still be rerun by ``run_pipeline.py`` and pass every gate.
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
    MLP_MIN_TRAIN_MONTHS,
    PERCENTILE_BREAKS,
    PERCENTILE_MIN_HISTORY,
    PERCENTILE_WEIGHTS,
    POSITION_SIZING_POLICIES,
    RAW_TOP_FEATURES_PER_BASE,
    RANDOM_SEED,
    SINGLE_FACTOR_MODEL_TYPE,
    SINGLE_FACTOR_REFIT_EVERY,
    SMOOTHED_LINEAR_SPAN,
    STATIC_FALLBACK_WEIGHT,
    TOP_FEATURE_POOL,
    TRANSACTION_COST_BPS,
    TRANSACTION_COST_SCENARIOS_BPS,
)
from src.modeling import evaluate_probabilities
from src.validation import (
    compare_position_sizing,
    evaluate_signal_gate,
    make_purged_outer_folds,
    nested_outer_predict,
    select_position_policy,
)


SPECS = {
    "adam_balanced_mlp_select": {
        "selection_model_type": "mlp",
        "params": {
            "hidden_layer_sizes": (8, 4),
            "activation": "tanh",
            "solver": "adam",
            "alpha": 0.05,
            "max_iter": 500,
            "tol": 1e-3,
            "learning_rate_init": 0.001,
            "batch_size": 32,
            "shuffle": False,
            "n_iter_no_change": 30,
            "balance_classes": True,
        },
    },
    "adam_balanced_logistic_select": {
        "selection_model_type": "logistic",
        "params": {
            "hidden_layer_sizes": (8, 4),
            "activation": "tanh",
            "solver": "adam",
            "alpha": 0.05,
            "max_iter": 500,
            "tol": 1e-3,
            "learning_rate_init": 0.001,
            "batch_size": 32,
            "shuffle": False,
            "n_iter_no_change": 30,
            "balance_classes": True,
        },
    },
    "lbfgs4_a1_balanced_logistic_select": {
        "selection_model_type": "logistic",
        "params": {
            "hidden_layer_sizes": (4,),
            "activation": "tanh",
            "solver": "lbfgs",
            "alpha": 1.0,
            "max_iter": 1000,
            "tol": 1e-4,
            "shuffle": False,
            "balance_classes": True,
        },
    },
    "adam_mlp_select": {
        "selection_model_type": "mlp",
        "params": {
            "hidden_layer_sizes": (8, 4),
            "activation": "tanh",
            "solver": "adam",
            "alpha": 0.05,
            "max_iter": 500,
            "tol": 1e-3,
            "learning_rate_init": 0.001,
            "batch_size": 32,
            "shuffle": False,
            "n_iter_no_change": 30,
        },
    },
    "adam_logistic_select": {
        "selection_model_type": "logistic",
        "params": {
            "hidden_layer_sizes": (8, 4),
            "activation": "tanh",
            "solver": "adam",
            "alpha": 0.05,
            "max_iter": 500,
            "tol": 1e-3,
            "learning_rate_init": 0.001,
            "batch_size": 32,
            "shuffle": False,
            "n_iter_no_change": 30,
        },
    },
    "lbfgs4_a1_logistic_select": {
        "selection_model_type": "logistic",
        "params": {
            "hidden_layer_sizes": (4,),
            "activation": "tanh",
            "solver": "lbfgs",
            "alpha": 1.0,
            "max_iter": 1000,
            "tol": 1e-4,
            "shuffle": False,
        },
    },
    "lbfgs2_a10_logistic_select": {
        "selection_model_type": "logistic",
        "params": {
            "hidden_layer_sizes": (2,),
            "activation": "tanh",
            "solver": "lbfgs",
            "alpha": 10.0,
            "max_iter": 1000,
            "tol": 1e-4,
            "shuffle": False,
        },
    },
}


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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--spec", required=True, choices=sorted(SPECS))
    parser.add_argument(
        "--inner-validation-months",
        type=int,
        help="override the stored fold design using a longer inner window",
    )
    parser.add_argument(
        "--final-max-train-months",
        type=int,
        help="optional causal rolling window for each outer MLP fit",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (run_dir / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    holdout_start = pd.Timestamp(manifest["config"]["research_holdout_start"])
    research_end = holdout_start - pd.offsets.MonthEnd(1)
    X = pd.read_parquet(run_dir / "factor_matrix.parquet").loc[:research_end]
    target = pd.read_csv(run_dir / "target.csv", index_col=0, parse_dates=True)
    target = target.loc[:research_end]
    candidates = pd.read_csv(
        run_dir / "mlp_deployment_candidate_universe.csv"
    )["feature"].tolist()
    candidate_metadata = pd.read_csv(run_dir / "factor_candidates.csv")
    feature_groups = (
        candidate_metadata.drop_duplicates("base").set_index("base")["group"].to_dict()
    )
    if args.inner_validation_months:
        folds = make_purged_outer_folds(
            target["y"].dropna().index,
            research_end=research_end,
            min_train_months=MLP_MIN_TRAIN_MONTHS,
            inner_validation_months=args.inner_validation_months,
            outer_validation_months=36,
            purge_months=FORECAST_HORIZON,
            screening_oos_months=MIN_OOS_PREDICTIONS,
        )
    else:
        fold_frame = pd.read_csv(
            run_dir / "mlp_outer_validation_folds.csv",
            parse_dates=[
                "development_end",
                "inner_validation_start",
                "inner_validation_end",
                "outer_start",
                "outer_end",
            ],
        )
        folds = [
            SimpleNamespace(**row._asdict())
            for row in fold_frame.itertuples(index=False)
        ]
    definition = SPECS[args.spec]
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    prediction, selections, screening = nested_outer_predict(
        X=X[candidates],
        y=target["y"],
        folds=folds,
        feature_groups=feature_groups,
        screening_model_type=SINGLE_FACTOR_MODEL_TYPE,
        selection_model_type=definition["selection_model_type"],
        final_model_type="mlp",
        min_train_months=MLP_MIN_TRAIN_MONTHS,
        final_min_train_months=MLP_MIN_TRAIN_MONTHS,
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
        svm_params={},
        mlp_params=definition["params"],
        calibration_splits=3,
        random_state=RANDOM_SEED,
        combination_refit_every=COMBINATION_SELECTION_REFIT_EVERY,
        final_refit_every=1,
        allow_unavailable_folds=True,
        final_max_train_months=args.final_max_train_months,
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
    panel = pd.read_parquet(run_dir / "monthly_panel.parquet")
    price = pd.read_csv(
        run_dir / "portfolio_return_source_monthly.csv",
        index_col=0,
        parse_dates=True,
    ).iloc[:, 0]
    labels = pd.Series(np.nan, index=prediction.index)
    for fold in gate_folds:
        labels.loc[fold.outer_start : fold.outer_end] = fold.fold
    comparison, _, fold_results = compare_position_sizing(
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
    summary = {
        "spec": args.spec,
        "selection_model_type": definition["selection_model_type"],
        "params_json": json.dumps(definition["params"], sort_keys=True),
        "completed_folds": len(gate_folds),
        "aggregate_auc": signal_summary["aggregate_auc"],
        "aggregate_rank_ic": signal_summary["aggregate_rank_ic"],
        "fold_joint_direction_pass_ratio": signal_summary[
            "fold_joint_direction_pass_ratio"
        ],
        "signal_gate_passed": signal_summary["signal_gate_passed"],
        "selected_policy": policy,
        "portfolio_gate_passed": bool(decision["portfolio_gate_passed"]),
        "median_fold_sharpe_difference": decision[
            "median_fold_Sharpe_difference"
        ],
        "annualized_active_return_10bps": decision[
            "annualized_active_return_10bps"
        ],
        "historical_holdout_opened": False,
        "inner_validation_months": args.inner_validation_months or 24,
        "final_max_train_months": args.final_max_train_months,
        "promotion_use": False,
    }
    prefix = output_dir / args.spec
    pd.DataFrame([summary]).to_csv(f"{prefix}_summary.csv", index=False)
    prediction.to_csv(f"{prefix}_prediction.csv")
    selections.to_csv(f"{prefix}_selections.csv", index=False)
    screening.to_csv(f"{prefix}_screening.csv", index=False)
    fold_details.to_csv(f"{prefix}_fold_signal.csv", index=False)
    decisions.to_csv(f"{prefix}_policy_gate.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
