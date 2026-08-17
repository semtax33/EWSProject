"""Compare small-sample MLP specifications without opening the holdout.

This script is deliberately limited to dates ending before the configured
historical research holdout.  It is an architecture research tool, not a
promotion shortcut: a winning row still has to be rerun through the full
nested pipeline and all operational gates.
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

from run_pipeline import _fold_signal_gate
from src.analytics import compute_return_ic
from src.config import (
    FIXED_BIN_THRESHOLDS,
    FIXED_BIN_WEIGHTS,
    MAX_STOCK_WEIGHT,
    MIN_STOCK_WEIGHT,
    PERCENTILE_BREAKS,
    PERCENTILE_MIN_HISTORY,
    PERCENTILE_WEIGHTS,
    POSITION_SIZING_POLICIES,
    SMOOTHED_LINEAR_SPAN,
    STATIC_FALLBACK_WEIGHT,
    TRANSACTION_COST_SCENARIOS_BPS,
)
from src.modeling import evaluate_probabilities, walk_forward_predict
from src.validation import (
    compare_position_sizing,
    evaluate_signal_gate,
    select_position_policy,
)


MLP_RESEARCH_SPECS = (
    {
        "spec": "adam_core4",
        "feature_set": "core4",
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
    {
        "spec": "lbfgs4_core4",
        "feature_set": "core4",
        "params": {
            "hidden_layer_sizes": (4,),
            "activation": "tanh",
            "solver": "lbfgs",
            "alpha": 0.10,
            "max_iter": 1000,
            "tol": 1e-4,
            "shuffle": False,
        },
    },
    {
        "spec": "lbfgs4_core6",
        "feature_set": "core6",
        "params": {
            "hidden_layer_sizes": (4,),
            "activation": "tanh",
            "solver": "lbfgs",
            "alpha": 0.10,
            "max_iter": 1000,
            "tol": 1e-4,
            "shuffle": False,
        },
    },
    {
        "spec": "lbfgs8_core6",
        "feature_set": "core6",
        "params": {
            "hidden_layer_sizes": (8,),
            "activation": "tanh",
            "solver": "lbfgs",
            "alpha": 0.10,
            "max_iter": 1000,
            "tol": 1e-4,
            "shuffle": False,
        },
    },
    {
        "spec": "adam4_core6",
        "feature_set": "core6",
        "params": {
            "hidden_layer_sizes": (4,),
            "activation": "tanh",
            "solver": "adam",
            "alpha": 0.10,
            "max_iter": 750,
            "tol": 1e-3,
            "learning_rate_init": 0.001,
            "batch_size": 32,
            "shuffle": False,
            "n_iter_no_change": 40,
        },
    },
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-holdout-only MLP architecture research"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--refit-every",
        type=int,
        default=1,
        help="research cadence; 1 matches the frozen monthly scoring protocol",
    )
    return parser.parse_args()


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


def run_research(run_dir: Path, refit_every: int):
    manifest = json.loads(
        (run_dir / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    prefix = manifest["config"]["market_series"]
    holdout_start = pd.Timestamp(manifest["config"]["research_holdout_start"])
    research_end = holdout_start - pd.offsets.MonthEnd(1)

    X = pd.read_parquet(run_dir / "factor_matrix.parquet")
    target = pd.read_csv(
        run_dir / "target.csv", index_col=0, parse_dates=True
    )
    panel = pd.read_parquet(run_dir / "monthly_panel.parquet")
    market_price = pd.read_csv(
        run_dir / "portfolio_return_source_monthly.csv",
        index_col=0,
        parse_dates=True,
    ).iloc[:, 0]
    folds = _load_folds(run_dir)

    core4 = [
        "term_spread_10y2y__level",
        f"{prefix}_trading_volume_ratio_12m",
        f"{prefix}_realized_volatility_1m",
        f"{prefix}_downside_volatility_1m",
    ]
    feature_sets = {
        "core4": core4,
        "core6": core4 + [f"{prefix}_momentum_6m", f"{prefix}_trend_10m"],
    }
    missing = sorted(
        feature
        for features in feature_sets.values()
        for feature in features
        if feature not in X
    )
    if missing:
        raise KeyError(f"MLP research features are missing: {missing}")

    rows = []
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for definition in MLP_RESEARCH_SPECS:
        name = definition["spec"]
        features = feature_sets[definition["feature_set"]]
        print(f"[MLP RESEARCH] {name}", flush=True)
        prediction = walk_forward_predict(
            X[features],
            target["y"],
            eval_start=folds[0].outer_start,
            eval_end=research_end,
            min_train=84,
            purge=int(manifest["config"]["forecast_horizon_months"]),
            refit_every=refit_every,
            model_type="mlp",
            mlp_params=definition["params"],
            random_state=int(manifest["config"]["random_seed"]),
        )
        metrics = evaluate_probabilities(prediction, target["y"])
        ic_summary, _, _ = compute_return_ic(
            prediction, target["future_return"], rolling_window=36
        )
        fold_input = _fold_signal_gate(
            folds, prediction, target["y"], target["future_return"]
        )
        signal_summary, _ = evaluate_signal_gate(
            fold_input,
            aggregate_auc=metrics["auc"],
            aggregate_rank_ic=ic_summary["rank_ic"],
        )
        fold_labels = pd.Series(np.nan, index=prediction.index)
        for fold in folds:
            fold_labels.loc[fold.outer_start : fold.outer_end] = fold.fold
        comparison, _, fold_results = compare_position_sizing(
            market_price=market_price,
            raw_ews=(prediction * 100).rename("raw_ews"),
            cash_yield=panel["cash_yield_3m"],
            policies=POSITION_SIZING_POLICIES,
            transaction_cost_scenarios=TRANSACTION_COST_SCENARIOS_BPS,
            sizing_config=_sizing_config(),
            fold_labels=fold_labels,
            evaluation_end=research_end,
        )
        policy, decisions = select_position_policy(
            comparison, fold_results, baseline="static_50_50"
        )
        decision = decisions.loc[decisions["policy"].eq(policy)].iloc[0]
        rows.append(
            {
                "spec": name,
                "feature_set": definition["feature_set"],
                "features": "|".join(features),
                "refit_every": refit_every,
                "observations": int(metrics["n"]),
                "aggregate_auc": metrics["auc"],
                "aggregate_rank_ic": ic_summary["rank_ic"],
                "fold_evaluable_ratio": signal_summary["fold_evaluable_ratio"],
                "fold_joint_direction_pass_ratio": signal_summary[
                    "fold_joint_direction_pass_ratio"
                ],
                "signal_gate_passed": signal_summary["signal_gate_passed"],
                "selected_policy": policy,
                "portfolio_gate_passed": decision["portfolio_gate_passed"],
                "median_fold_sharpe_difference": decision[
                    "median_fold_Sharpe_difference"
                ],
                "annualized_active_return_10bps": decision[
                    "annualized_active_return_10bps"
                ],
                "params_json": json.dumps(definition["params"], sort_keys=True),
                "selection_use": "pre_holdout_architecture_research_only",
                "holdout_opened": False,
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output).resolve()
    if args.refit_every < 1:
        raise ValueError("refit-every must be positive")
    result = run_research(run_dir, args.refit_every)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(output)


if __name__ == "__main__":
    main()
