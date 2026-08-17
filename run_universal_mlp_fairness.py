"""Run an apples-to-apples three-market MLP portability evaluation.

This track deliberately does not reuse the market-specific KOSPI, S&P 500 or
NASDAQ-100 target/model/allocation choices.  Every market receives the same
four exact input columns, cash-excess target, autonomous MLP hyperparameters,
expanding-percentile allocation, fold calendar and transaction-cost gates.

Two claims are tested separately:

1. ``same_spec``: identical code/specification, fitted independently per market.
2. ``leave_one_market_out``: the held-out market contributes no labels; one MLP
   is fitted on the other two markets' histories and transferred unchanged.

All model and policy decisions are evaluated on data ending 2020-03.  The
2020-04+ period is reported only as a fail-closed historical safety diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

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
    RANDOM_SEED,
    SMOOTHED_LINEAR_SPAN,
    STATIC_FALLBACK_WEIGHT,
)
from src.modeling import (
    build_model_target,
    evaluate_probabilities,
    fit_classification_model,
    make_model,
    walk_forward_predict,
)
from src.validation import (
    compare_position_sizing,
    evaluate_holdout_safety_veto,
    evaluate_signal_gate,
    select_position_policy,
)


UNIVERSAL_FEATURES = (
    "us_corporate_equity_value__dist_ma_3m",
    "term_spread_10y3m__ma_60m_chg_2m",
    "usd_per_aud__ma_12m_chg_6m",
    "us_nonfinancial_profits_after_tax__vol_6m",
)

# This is the S&P specification that had already been locked on pre-2020 data
# before this portability test.  No market is allowed to override a parameter.
UNIVERSAL_MLP_PARAMS = {
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

TARGET_MODE = "cash_excess"
ALLOCATION_POLICY = "expanding_percentile"
POLICIES = (ALLOCATION_POLICY, "static_50_50")
TRANSACTION_COST_SCENARIOS = (10, 25)
FORECAST_HORIZON = 3
MIN_TRAIN_MONTHS = 84
REFIT_EVERY = 1
RESEARCH_END = pd.Timestamp("2020-03-31")
HOLDOUT_START = pd.Timestamp("2020-04-30")
HOLDOUT_END = pd.Timestamp("2026-05-31")

# A common calendar is required for a meaningful cross-market fold comparison.
COMMON_FOLDS = (
    SimpleNamespace(fold=1, outer_start=pd.Timestamp("2008-11-30"), outer_end=pd.Timestamp("2011-10-31")),
    SimpleNamespace(fold=2, outer_start=pd.Timestamp("2011-11-30"), outer_end=pd.Timestamp("2014-10-31")),
    SimpleNamespace(fold=3, outer_start=pd.Timestamp("2014-11-30"), outer_end=pd.Timestamp("2017-10-31")),
    SimpleNamespace(fold=4, outer_start=pd.Timestamp("2017-11-30"), outer_end=RESEARCH_END),
)


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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kospi-run", required=True)
    parser.add_argument("--sp500-run", required=True)
    parser.add_argument("--nasdaq-run", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_market(run_dir: str | Path):
    run_dir = Path(run_dir).resolve()
    manifest = json.loads(
        (run_dir / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    config = manifest["config"]
    X = pd.read_parquet(run_dir / "factor_matrix.parquet")[
        list(UNIVERSAL_FEATURES)
    ].copy()
    signal_price = pd.read_csv(
        run_dir / "market_monthly.csv", index_col=0, parse_dates=True
    ).iloc[:, 0]
    portfolio_price = pd.read_csv(
        run_dir / "portfolio_return_source_monthly.csv",
        index_col=0,
        parse_dates=True,
    ).iloc[:, 0]
    panel = pd.read_parquet(run_dir / "monthly_panel.parquet")
    cash_yield = panel["cash_yield_3m"]
    target = build_model_target(
        signal_price,
        mode=TARGET_MODE,
        horizon=FORECAST_HORIZON,
        cash_yield=cash_yield,
    )
    return {
        "run_dir": run_dir,
        "key": config["market_key"],
        "name": config["market_name"],
        "X": X,
        "target": target,
        "portfolio_price": portfolio_price,
        "cash_yield": cash_yield,
        "code_files": manifest["code_files"],
    }


def validate_common_inputs(markets):
    """Fail if the supposedly universal feature history differs by market."""
    reference = markets[0]["X"]
    for market in markets[1:]:
        pd.testing.assert_frame_equal(
            reference,
            market["X"],
            check_exact=True,
            obj=f"universal feature matrix: {market['key']}",
        )
    hashes = {
        hashlib.sha256(
            json.dumps(market["code_files"], sort_keys=True).encode("utf-8")
        ).hexdigest()
        for market in markets
    }
    if len(hashes) != 1:
        raise AssertionError("Input runs do not share one code-files manifest hash")
    return next(iter(hashes))


def pooled_walk_forward_predict(
    *,
    X_test,
    source_markets,
    eval_start,
    eval_end,
    min_train=MIN_TRAIN_MONTHS,
    purge=FORECAST_HORIZON,
    refit_every=REFIT_EVERY,
    mlp_params=None,
    random_state=RANDOM_SEED,
):
    """Predict one held-out market without using any of its labels."""
    X_test = X_test.sort_index()
    eval_dates = X_test.loc[pd.Timestamp(eval_start) : pd.Timestamp(eval_end)].index
    prediction = pd.Series(
        np.nan, index=eval_dates, dtype=float, name="leave_one_market_out_prediction"
    )
    model = None
    for i, test_date in enumerate(eval_dates):
        x_test = X_test.loc[[test_date]]
        if x_test.isna().any(axis=None):
            continue
        if model is None or i % refit_every == 0:
            cutoff = (test_date.to_period("M") - purge).to_timestamp("M")
            frames = []
            source_ready = True
            for source in source_markets:
                train = pd.concat(
                    [
                        source["X"].loc[:cutoff],
                        source["target"]["y"].loc[:cutoff].rename("y"),
                    ],
                    axis=1,
                ).dropna()
                if len(train) < min_train or train["y"].nunique() < 2:
                    source_ready = False
                    break
                train = train.assign(source_market=source["key"])
                frames.append(train)
            if not source_ready:
                model = None
                continue
            pooled = pd.concat(frames, axis=0, ignore_index=True)
            model = make_model(
                "mlp",
                mlp_params=dict(mlp_params or UNIVERSAL_MLP_PARAMS),
                random_state=random_state,
            )
            fit_classification_model(
                model, pooled[list(UNIVERSAL_FEATURES)], pooled["y"]
            )
        prediction.loc[test_date] = model.predict_proba(x_test)[0, 1]
    return prediction


def independently_fitted_prediction(market):
    return walk_forward_predict(
        market["X"],
        market["target"]["y"],
        eval_start=COMMON_FOLDS[0].outer_start,
        eval_end=HOLDOUT_END,
        min_train=MIN_TRAIN_MONTHS,
        purge=FORECAST_HORIZON,
        refit_every=REFIT_EVERY,
        model_type="mlp",
        mlp_params=UNIVERSAL_MLP_PARAMS,
        random_state=RANDOM_SEED,
    )


def evaluate_track(market, prediction, *, track, output_dir):
    research_prediction = prediction.loc[:RESEARCH_END]
    metrics = evaluate_probabilities(research_prediction, market["target"]["y"])
    ic, _, _ = compute_return_ic(
        research_prediction, market["target"]["future_return"], rolling_window=36
    )
    fold_signal = _fold_signal_gate(
        COMMON_FOLDS,
        research_prediction,
        market["target"]["y"],
        market["target"]["future_return"],
    )
    signal_summary, signal_details = evaluate_signal_gate(
        fold_signal,
        aggregate_auc=metrics["auc"],
        aggregate_rank_ic=ic["rank_ic"],
    )
    fold_labels = pd.Series(np.nan, index=research_prediction.index)
    for fold in COMMON_FOLDS:
        fold_labels.loc[fold.outer_start : fold.outer_end] = fold.fold
    comparison, monthly, fold_results = compare_position_sizing(
        market_price=market["portfolio_price"],
        raw_ews=(prediction * 100).rename("raw_ews"),
        cash_yield=market["cash_yield"],
        policies=POLICIES,
        transaction_cost_scenarios=TRANSACTION_COST_SCENARIOS,
        sizing_config=sizing_config(),
        fold_labels=fold_labels,
        evaluation_end=RESEARCH_END,
    )
    _, decisions = select_position_policy(
        comparison, fold_results, baseline="static_50_50"
    )
    policy_decision = decisions.loc[
        decisions["policy"].eq(ALLOCATION_POLICY)
    ].iloc[0]

    holdout_prediction = prediction.loc[HOLDOUT_START:HOLDOUT_END]
    holdout_metrics = evaluate_probabilities(
        holdout_prediction, market["target"]["y"]
    )
    holdout_ic, _, _ = compute_return_ic(
        holdout_prediction,
        market["target"]["future_return"],
        rolling_window=36,
    )
    holdout_comparison, holdout_monthly, _ = compare_position_sizing(
        market_price=market["portfolio_price"],
        raw_ews=(prediction * 100).rename("raw_ews"),
        cash_yield=market["cash_yield"],
        policies=POLICIES,
        transaction_cost_scenarios=(10,),
        sizing_config=sizing_config(),
        evaluation_start=HOLDOUT_START,
        evaluation_end=HOLDOUT_END,
    )
    holdout_policy = holdout_comparison.loc[
        holdout_comparison["policy"].eq(ALLOCATION_POLICY)
        & holdout_comparison["transaction_cost_bps"].eq(10)
    ].iloc[0]
    holdout_safety, holdout_checks, holdout_differences = (
        evaluate_holdout_safety_veto(
            auc=holdout_metrics["auc"],
            rank_ic=holdout_ic["rank_ic"],
            dynamic_sharpe=holdout_policy["Sharpe"],
            same_exposure_sharpe=holdout_policy["same_exposure_Sharpe"],
            dynamic_cagr=holdout_policy["CAGR"],
            same_exposure_cagr=holdout_policy["same_exposure_CAGR"],
            dynamic_max_drawdown=holdout_policy["MaxDrawdown"],
            same_exposure_max_drawdown=holdout_policy[
                "same_exposure_MaxDrawdown"
            ],
        )
    )
    joint_pre2020 = bool(
        signal_summary["signal_gate_passed"]
        and policy_decision["portfolio_gate_passed"]
    )
    row = {
        "track": track,
        "market_key": market["key"],
        "market_name": market["name"],
        "same_exact_features": True,
        "same_target": TARGET_MODE,
        "same_model_params": True,
        "same_refit_every_months": REFIT_EVERY,
        "same_min_train_months": MIN_TRAIN_MONTHS,
        "same_allocation_policy": ALLOCATION_POLICY,
        "same_transaction_cost_scenarios": "10|25",
        "pre2020_auc": signal_summary["aggregate_auc"],
        "pre2020_rank_ic": signal_summary["aggregate_rank_ic"],
        "pre2020_fold_joint_direction_ratio": signal_summary[
            "fold_joint_direction_pass_ratio"
        ],
        "pre2020_signal_gate": bool(signal_summary["signal_gate_passed"]),
        "pre2020_median_fold_sharpe_difference": policy_decision[
            "median_fold_Sharpe_difference"
        ],
        "pre2020_positive_fold_ratio": policy_decision["positive_fold_ratio"],
        "pre2020_active_return_25bps": policy_decision[
            "annualized_active_return_25bps"
        ],
        "pre2020_drawdown_difference_25bps": policy_decision[
            "drawdown_difference_25bps"
        ],
        "pre2020_portfolio_gate": bool(
            policy_decision["portfolio_gate_passed"]
        ),
        "pre2020_joint_gate": joint_pre2020,
        "holdout_auc": holdout_metrics["auc"],
        "holdout_rank_ic": holdout_ic["rank_ic"],
        "holdout_dynamic_sharpe": holdout_policy["Sharpe"],
        "holdout_same_exposure_sharpe": holdout_policy[
            "same_exposure_Sharpe"
        ],
        "holdout_sharpe_difference": holdout_differences["sharpe_difference"],
        "holdout_dynamic_cagr": holdout_policy["CAGR"],
        "holdout_same_exposure_cagr": holdout_policy["same_exposure_CAGR"],
        "holdout_cagr_difference": holdout_differences["cagr_difference"],
        "holdout_dynamic_max_drawdown": holdout_policy["MaxDrawdown"],
        "holdout_same_exposure_max_drawdown": holdout_policy[
            "same_exposure_MaxDrawdown"
        ],
        "holdout_drawdown_difference": holdout_differences[
            "drawdown_difference"
        ],
        **{f"holdout_{key}": value for key, value in holdout_checks.items()},
        "holdout_safety_gate": bool(holdout_safety),
        "fully_validated_historical": bool(joint_pre2020 and holdout_safety),
        "holdout_used_for_selection": False,
        "interpretation": (
            "portable_under_declared_historical_gates"
            if joint_pre2020 and holdout_safety
            else "portability_not_demonstrated"
        ),
    }
    stem = output_dir / f"{track}_{market['key']}"
    prediction.to_csv(f"{stem}_prediction.csv")
    signal_details.to_csv(f"{stem}_pre2020_signal_folds.csv", index=False)
    comparison.to_csv(f"{stem}_pre2020_policy_comparison.csv", index=False)
    fold_results.to_csv(f"{stem}_pre2020_policy_folds.csv", index=False)
    decisions.to_csv(f"{stem}_pre2020_policy_gate.csv", index=False)
    monthly.to_csv(f"{stem}_pre2020_policy_monthly.csv")
    holdout_comparison.to_csv(f"{stem}_holdout_comparison.csv", index=False)
    holdout_monthly.to_csv(f"{stem}_holdout_monthly.csv")
    return row


def run(run_dirs, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    markets = [load_market(path) for path in run_dirs]
    code_manifest_hash = validate_common_inputs(markets)
    protocol = {
        "schema_version": 1,
        "status": "diagnostic_research_only",
        "claim_tested": (
            "one_identical_mlp_specification_is_portable_across_three_markets"
        ),
        "features": list(UNIVERSAL_FEATURES),
        "target_mode": TARGET_MODE,
        "forecast_horizon_months": FORECAST_HORIZON,
        "model_type": "autonomous_mlp",
        "model_params": {
            **UNIVERSAL_MLP_PARAMS,
            "hidden_layer_sizes": list(
                UNIVERSAL_MLP_PARAMS["hidden_layer_sizes"]
            ),
        },
        "minimum_training_months_per_source_market": MIN_TRAIN_MONTHS,
        "refit_every_months": REFIT_EVERY,
        "allocation_policy": ALLOCATION_POLICY,
        "transaction_cost_scenarios_bps": list(TRANSACTION_COST_SCENARIOS),
        "common_outer_folds": [
            {
                "fold": fold.fold,
                "outer_start": fold.outer_start.date().isoformat(),
                "outer_end": fold.outer_end.date().isoformat(),
            }
            for fold in COMMON_FOLDS
        ],
        "research_end": RESEARCH_END.date().isoformat(),
        "historical_holdout_start": HOLDOUT_START.date().isoformat(),
        "historical_holdout_end": HOLDOUT_END.date().isoformat(),
        "holdout_used_for_selection": False,
        "leave_one_market_out_uses_heldout_labels": False,
        "input_code_manifest_hash": code_manifest_hash,
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    same_spec_rows = []
    same_spec_predictions = {}
    for market in markets:
        print(f"[SAME SPEC] {market['name']}", flush=True)
        prediction = independently_fitted_prediction(market)
        same_spec_predictions[market["key"]] = prediction
        same_spec_rows.append(
            evaluate_track(
                market,
                prediction,
                track="same_spec",
                output_dir=output_dir,
            )
        )
    same_spec = pd.DataFrame(same_spec_rows)
    same_spec.to_csv(output_dir / "same_spec_summary.csv", index=False)

    lomo_rows = []
    for heldout in markets:
        sources = [market for market in markets if market["key"] != heldout["key"]]
        print(
            f"[LEAVE ONE MARKET OUT] {heldout['name']} <- "
            + ", ".join(source["name"] for source in sources),
            flush=True,
        )
        prediction = pooled_walk_forward_predict(
            X_test=heldout["X"],
            source_markets=sources,
            eval_start=COMMON_FOLDS[0].outer_start,
            eval_end=HOLDOUT_END,
            mlp_params=UNIVERSAL_MLP_PARAMS,
            random_state=RANDOM_SEED,
        )
        row = evaluate_track(
            heldout,
            prediction,
            track="leave_one_market_out",
            output_dir=output_dir,
        )
        row["training_markets"] = "|".join(source["key"] for source in sources)
        row["heldout_market_labels_used"] = False
        lomo_rows.append(row)
    lomo = pd.DataFrame(lomo_rows)
    lomo.to_csv(output_dir / "leave_one_market_out_summary.csv", index=False)

    same_spec_all = bool(same_spec["fully_validated_historical"].all())
    lomo_all = bool(lomo["fully_validated_historical"].all())
    conclusion = {
        "same_spec_all_three_fully_validated": same_spec_all,
        "leave_one_market_out_all_three_fully_validated": lomo_all,
        "universal_model_claim_supported": bool(same_spec_all and lomo_all),
        "market_specific_models_remain_separate_claim": True,
        "historical_holdout_is_not_fresh": True,
        "decision": (
            "universal_portability_supported_historically"
            if same_spec_all and lomo_all
            else "universal_portability_not_demonstrated;retain_market_specific_models"
        ),
    }
    (output_dir / "fairness_conclusion.json").write_text(
        json.dumps(conclusion, indent=2), encoding="utf-8"
    )
    print("\nSAME SPEC")
    print(same_spec.to_string(index=False))
    print("\nLEAVE ONE MARKET OUT")
    print(lomo.to_string(index=False))
    print("\n" + json.dumps(conclusion, indent=2))
    return conclusion


def main():
    args = parse_args()
    run(
        [args.kospi_run, args.sp500_run, args.nasdaq_run],
        Path(args.output_dir).resolve(),
    )


if __name__ == "__main__":
    main()
