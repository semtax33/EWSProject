"""Compare a small, predeclared set of fixed model specifications pre-holdout.

The research boundary is the month before ``research_holdout_start`` stored in
the source run manifest.  Candidate membership, estimator parameters and the
allocation-policy menu are declared in this file; no 2020+ observation is read
for scoring or selection.  The winning specification is only a candidate for
one subsequent holdout safety veto, never proof of deployment readiness.
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


SVM_PARAMS = {"C": 1.0, "kernel": "rbf", "gamma": "scale"}
SMALL_SAMPLE_MLP_PARAMS = {
    "hidden_layer_sizes": (4,),
    "activation": "tanh",
    "solver": "lbfgs",
    "alpha": 0.10,
    "max_iter": 1000,
    "tol": 1e-4,
    "shuffle": False,
}
TRANSFERRED_US_EQUITY_MLP_PARAMS = {
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
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        help="run only the named predeclared candidate; repeat to select several",
    )
    parser.add_argument(
        "--target-mode",
        choices=(
            "absolute_positive",
            "cash_excess",
            "future_trend_10m",
            "future_drawdown_5pct",
        ),
        default="absolute_positive",
        help=(
            "cash_excess labels a month risk-on only when the forward index "
            "return exceeds the causal current-cash-yield hurdle"
        ),
    )
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


def _candidate_registry(prefix: str):
    market_structure = (
        "term_spread_10y2y__level",
        f"{prefix}_trading_volume_ratio_12m",
        f"{prefix}_realized_volatility_1m",
        f"{prefix}_downside_volatility_1m",
        f"{prefix}_momentum_6m",
        f"{prefix}_trend_10m",
    )
    trend_only = (f"{prefix}_trend_10m",)
    trend_momentum = (
        f"{prefix}_trend_10m",
        f"{prefix}_momentum_12m",
    )
    trend_risk = (
        f"{prefix}_trend_10m",
        f"{prefix}_momentum_12m",
        f"{prefix}_realized_volatility_1m",
        f"{prefix}_downside_volatility_1m",
    )
    # This complete feature specification was locked on the S&P 500's
    # pre-2020 development sample.  Applying it unchanged to another US
    # equity index is an external transfer test, not a NASDAQ holdout fit.
    us_equity_macro = (
        "us_corporate_equity_value__dist_ma_3m",
        "term_spread_10y3m__ma_60m_chg_2m",
        "usd_per_aud__ma_12m_chg_6m",
        "us_nonfinancial_profits_after_tax__vol_6m",
    )
    # Externally specified in the original KOSPI EWS material and kept fixed
    # independently of the 2020+ historical holdout.  These are the auditable
    # transforms already used by the locked KOSPI structural candidate.
    reference_kospi_structure = (
        "korea_stock_universe_trading_value_to_market_cap__ma_9m_chg_9m",
        "term_spread_10y2y__ma_48m_chg_3m",
        "korea_stock_universe_pairwise_correlation_1m__ewma_12m_chg_24m",
        "korea_stock_universe_return_skew_1m__ma_60m_chg_3m",
    )
    return (
        {
            "candidate": "trend_only1_logistic",
            "feature_set": "trend_only1",
            "features": trend_only,
            "model_type": "logistic",
            "model_params": {},
            "provenance": "predeclared_low_complexity_trend_v1",
        },
        {
            "candidate": "trend_momentum2_logistic",
            "feature_set": "trend_momentum2",
            "features": trend_momentum,
            "model_type": "logistic",
            "model_params": {},
            "provenance": "predeclared_low_complexity_trend_v1",
        },
        {
            "candidate": "trend_risk4_logistic",
            "feature_set": "trend_risk4",
            "features": trend_risk,
            "model_type": "logistic",
            "model_params": {},
            "provenance": "predeclared_low_complexity_trend_risk_v1",
        },
        {
            "candidate": "trend_risk4_svm",
            "feature_set": "trend_risk4",
            "features": trend_risk,
            "model_type": "svm",
            "model_params": SVM_PARAMS,
            "provenance": "predeclared_low_complexity_trend_risk_v1",
        },
        {
            "candidate": "market_structure6_logistic",
            "feature_set": "market_structure6",
            "features": market_structure,
            "model_type": "logistic",
            "model_params": {},
            "provenance": "predeclared_market_structure_v1",
        },
        {
            "candidate": "market_structure6_svm",
            "feature_set": "market_structure6",
            "features": market_structure,
            "model_type": "svm",
            "model_params": SVM_PARAMS,
            "provenance": "predeclared_market_structure_v1",
        },
        {
            "candidate": "market_structure6_mlp",
            "feature_set": "market_structure6",
            "features": market_structure,
            "model_type": "mlp",
            "model_params": SMALL_SAMPLE_MLP_PARAMS,
            "provenance": "predeclared_market_structure_v1",
        },
        {
            "candidate": "market_structure6_mlp_rolling120",
            "feature_set": "market_structure6",
            "features": market_structure,
            "model_type": "mlp",
            "model_params": SMALL_SAMPLE_MLP_PARAMS,
            "max_train_months": 120,
            "provenance": "predeclared_market_structure_rolling_window_v1",
        },
        {
            "candidate": "market_structure6_mlp_rolling180",
            "feature_set": "market_structure6",
            "features": market_structure,
            "model_type": "mlp",
            "model_params": SMALL_SAMPLE_MLP_PARAMS,
            "max_train_months": 180,
            "provenance": "predeclared_market_structure_rolling_window_v1",
        },
        {
            "candidate": "transferred_us_equity4_logistic",
            "feature_set": "transferred_us_equity4",
            "features": us_equity_macro,
            "model_type": "logistic",
            "model_params": {},
            "provenance": "locked_sp500_pre2020_transfer_v1",
        },
        {
            "candidate": "transferred_us_equity4_svm",
            "feature_set": "transferred_us_equity4",
            "features": us_equity_macro,
            "model_type": "svm",
            "model_params": SVM_PARAMS,
            "provenance": "locked_sp500_pre2020_transfer_v1",
        },
        {
            "candidate": "transferred_us_equity4_mlp",
            "feature_set": "transferred_us_equity4",
            "features": us_equity_macro,
            "model_type": "mlp",
            "model_params": TRANSFERRED_US_EQUITY_MLP_PARAMS,
            "provenance": "locked_sp500_pre2020_full_spec_transfer_v1",
        },
        {
            "candidate": "reference_kospi_structure4_logistic",
            "feature_set": "reference_kospi_structure4",
            "features": reference_kospi_structure,
            "model_type": "logistic",
            "model_params": {},
            "provenance": "original_ews_reference_kospi_structure_v1",
        },
        {
            "candidate": "reference_kospi_structure4_svm",
            "feature_set": "reference_kospi_structure4",
            "features": reference_kospi_structure,
            "model_type": "svm",
            "model_params": SVM_PARAMS,
            "provenance": "original_ews_reference_kospi_structure_v1",
        },
        {
            "candidate": "reference_kospi_structure4_mlp",
            "feature_set": "reference_kospi_structure4",
            "features": reference_kospi_structure,
            "model_type": "mlp",
            "model_params": SMALL_SAMPLE_MLP_PARAMS,
            "provenance": "original_ews_reference_kospi_structure_v1",
        },
    )


def _runtime_kwargs(candidate):
    model_type = candidate["model_type"]
    if model_type == "svm":
        return {
            "svm_params": candidate["model_params"],
            "mlp_params": None,
            "calibration_splits": 3,
        }
    if model_type == "mlp":
        return {
            "svm_params": None,
            "mlp_params": candidate["model_params"],
            "calibration_splits": 3,
        }
    return {"svm_params": None, "mlp_params": None, "calibration_splits": 3}


def run_research(
    run_dir: Path,
    output_dir: Path,
    *,
    target_mode: str,
    selected_candidates: list[str] | None = None,
):
    manifest = json.loads(
        (run_dir / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    prefix = manifest["config"]["market_series"]
    holdout_start = pd.Timestamp(manifest["config"]["research_holdout_start"])
    research_end = holdout_start - pd.offsets.MonthEnd(1)
    horizon = int(manifest["config"]["forecast_horizon_months"])
    if horizon != FORECAST_HORIZON:
        raise ValueError("Stored forecast horizon does not match the research code")

    X = pd.read_parquet(run_dir / "factor_matrix.parquet").loc[:research_end]
    target = pd.read_csv(run_dir / "target.csv", index_col=0, parse_dates=True)
    target = target.loc[:research_end]
    panel = pd.read_parquet(run_dir / "monthly_panel.parquet").loc[:research_end]
    price = pd.read_csv(
        run_dir / "portfolio_return_source_monthly.csv",
        index_col=0,
        parse_dates=True,
    ).iloc[:, 0].loc[:research_end]
    folds = _load_folds(run_dir)
    if target_mode == "cash_excess":
        cash_hurdle = (
            (1.0 + panel["cash_yield_3m"].clip(lower=-99.0) / 100.0)
            ** (horizon / 12.0)
            - 1.0
        )
        valid_target = target["future_return"].notna() & cash_hurdle.notna()
        research_y = pd.Series(np.nan, index=target.index, name="y")
        research_y.loc[valid_target] = (
            target.loc[valid_target, "future_return"]
            > cash_hurdle.loc[valid_target]
        ).astype(float)
    elif target_mode in {"future_trend_10m", "future_drawdown_5pct"}:
        signal_price = pd.read_csv(
            run_dir / "market_monthly.csv", index_col=0, parse_dates=True
        ).iloc[:, 0].loc[:research_end]
        if target_mode == "future_trend_10m":
            future_state = (
                signal_price > signal_price.rolling(10, min_periods=10).mean()
            ).shift(-horizon)
        else:
            forward_path_returns = pd.concat(
                [signal_price.shift(-step) / signal_price - 1.0 for step in range(1, horizon + 1)],
                axis=1,
            )
            future_state = (forward_path_returns.min(axis=1) > -0.05).astype(float)
            future_state[forward_path_returns.isna().any(axis=1)] = np.nan
        valid_target = target["future_return"].notna() & future_state.notna()
        research_y = pd.Series(np.nan, index=target.index, name="y")
        research_y.loc[valid_target] = future_state.loc[valid_target].astype(float)
    else:
        research_y = target["y"].copy()
    feature_metadata = pd.read_csv(run_dir / "factor_candidates.csv")
    deployment_features = set(
        feature_metadata.loc[
            feature_metadata["eligible_mlp_deployment_track"].astype(bool), "feature"
        ]
    )
    feature_groups = (
        feature_metadata.drop_duplicates("base").set_index("base")["group"].to_dict()
    )

    candidates = _candidate_registry(prefix)
    if selected_candidates:
        requested = set(selected_candidates)
        known = {row["candidate"] for row in candidates}
        unknown = sorted(requested.difference(known))
        if unknown:
            raise ValueError(f"Unknown fixed candidates: {unknown}")
        candidates = tuple(row for row in candidates if row["candidate"] in requested)
    all_features = {feature for row in candidates for feature in row["features"]}
    missing = sorted(all_features.difference(X.columns))
    unsafe = sorted(all_features.difference(deployment_features))
    if missing or unsafe:
        raise ValueError(
            f"Fixed candidate audit failed: missing={missing}, not_deployment_safe={unsafe}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    registry_payload = []
    rows = []
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for candidate in candidates:
        name = candidate["candidate"]
        print(f"[FIXED PRE-HOLDOUT] {name}", flush=True)
        prediction, selections, _ = fixed_outer_predict(
            X=X,
            y=research_y,
            folds=folds,
            features=candidate["features"],
            feature_groups=feature_groups,
            final_model_type=candidate["model_type"],
            final_min_train_months=84,
            horizon=horizon,
            refit_every=1,
            random_state=RANDOM_SEED,
            final_max_train_months=candidate.get("max_train_months"),
            selection_note=(candidate["provenance"] + ";holdout_not_opened"),
            **_runtime_kwargs(candidate),
        )
        metrics = evaluate_probabilities(prediction, research_y)
        ic, _, _ = compute_return_ic(
            prediction, target["future_return"], rolling_window=36
        )
        fold_rows = _fold_signal_gate(
            folds,
            prediction,
            research_y,
            target["future_return"],
            eligibility_starts=_fold_eligibility_map(selections),
        )
        signal_summary, fold_details = evaluate_signal_gate(
            fold_rows,
            aggregate_auc=metrics["auc"],
            aggregate_rank_ic=ic["rank_ic"],
        )
        labels = pd.Series(np.nan, index=prediction.index)
        for fold in folds:
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
        signal_pass = bool(signal_summary["signal_gate_passed"])
        portfolio_pass = bool(decision["portfolio_gate_passed"])
        eligible = bool(signal_pass and portfolio_pass and policy != "static_50_50")
        rows.append(
            {
                "candidate": name,
                "target_mode": target_mode,
                "feature_set": candidate["feature_set"],
                "model_type": candidate["model_type"],
                "max_train_months": candidate.get("max_train_months"),
                "provenance": candidate["provenance"],
                "features": "|".join(candidate["features"]),
                "aggregate_auc": signal_summary["aggregate_auc"],
                "aggregate_rank_ic": signal_summary["aggregate_rank_ic"],
                "fold_joint_direction_pass_ratio": signal_summary[
                    "fold_joint_direction_pass_ratio"
                ],
                "signal_gate_passed": signal_pass,
                "selected_policy": policy,
                "portfolio_gate_passed": portfolio_pass,
                "median_fold_sharpe_difference": decision[
                    "median_fold_Sharpe_difference"
                ],
                "positive_fold_ratio": decision["positive_fold_ratio"],
                "annualized_active_return_10bps": decision[
                    "annualized_active_return_10bps"
                ],
                "drawdown_difference_10bps": decision[
                    "drawdown_difference_10bps"
                ],
                "preholdout_candidate_eligible": eligible,
                "historical_holdout_opened": False,
            }
        )
        stem = output_dir / name
        prediction.to_csv(f"{stem}_prediction.csv")
        selections.to_csv(f"{stem}_selections.csv", index=False)
        fold_details.to_csv(f"{stem}_fold_signal.csv", index=False)
        comparison.to_csv(f"{stem}_policy_comparison.csv", index=False)
        decisions.to_csv(f"{stem}_policy_gate.csv", index=False)
        monthly.to_csv(f"{stem}_policy_monthly.csv")
        fold_results.to_csv(f"{stem}_policy_folds.csv", index=False)
        registry_payload.append(
            {
                **candidate,
                "features": list(candidate["features"]),
                "research_end": research_end.date().isoformat(),
                "target_mode": target_mode,
                "historical_holdout_opened": False,
            }
        )

    summary = pd.DataFrame(rows)
    eligible = summary.loc[summary["preholdout_candidate_eligible"]].sort_values(
        [
            "median_fold_sharpe_difference",
            "fold_joint_direction_pass_ratio",
            "aggregate_rank_ic",
            "candidate",
        ],
        ascending=[False, False, False, True],
    )
    summary["selected_preholdout"] = False
    if not eligible.empty:
        winner = eligible.iloc[0]["candidate"]
        summary.loc[summary["candidate"].eq(winner), "selected_preholdout"] = True
    else:
        winner = None
    summary["selection_rule"] = (
        "signal_and_portfolio_gates_then_median_fold_sharpe;holdout_not_used"
    )
    summary.to_csv(output_dir / "pre2020_candidate_summary.csv", index=False)
    (output_dir / "candidate_registry.json").write_text(
        json.dumps(registry_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "selected_candidate.json").write_text(
        json.dumps(
            {
                "selected_candidate": winner,
                "research_end": research_end.date().isoformat(),
                "target_mode": target_mode,
                "historical_holdout_opened": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    return summary


def main():
    args = parse_args()
    run_research(
        Path(args.run_dir).resolve(),
        Path(args.output_dir).resolve(),
        target_mode=args.target_mode,
        selected_candidates=args.candidate,
    )


if __name__ == "__main__":
    main()
