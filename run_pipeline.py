import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics import (
    compute_return_ic,
    performance_stats,
    rolling_sharpe,
)
from src.audit import (
    data_freshness_audit,
    feature_group_coverage,
    market_return_role_registry,
    overlapping_target_diagnostics,
    point_in_time_audit,
    selected_point_in_time_audit,
)
from src.backtest import run_backtest
from src.config import (
    ALFRED_DIR,
    COMBO_CANDIDATE_POOL,
    COMBINATION_SELECTION_REFIT_EVERY,
    BOOTSTRAP_BLOCK_MONTHS,
    BOOTSTRAP_SAMPLES,
    CORRELATION_THRESHOLD,
    ECONOMIC_REVIEW_FILE,
    EXACT_INDICATOR_GAP_FILE,
    EXHAUSTIVE_COMBO_CANDIDATE_POOL,
    EXHAUSTIVE_MAX_COMBO_SIZE,
    EXTERNAL_REFERENCE_FILE,
    EWS_RISK_OFF,
    EWS_RISK_ON,
    FINAL_MODEL_TYPE,
    FIXED_BIN_THRESHOLDS,
    FIXED_BIN_WEIGHTS,
    FORECAST_HORIZON,
    FRED_DIR,
    GROUP_CANDIDATES_PER_GROUP,
    IC_ROLLING_WINDOW,
    KOREA_STOCK_PANEL_FILE,
    MARCAP_KOSPI200_FILE,
    MARCAP_KOSPI200_METADATA_FILE,
    MARKET_BREADTH_FILE,
    MARKET_BREADTH_METADATA_FILE,
    MARKET_NAME,
    MARKET_SERIES_NAME,
    MAX_FEATURES_PER_BASE,
    MAX_FEATURES_PER_GROUP,
    MAX_MODEL_FEATURES,
    MAX_STOCK_WEIGHT,
    MIN_DISTINCT_GROUPS,
    MIN_EXPANDED_RAW_SERIES,
    MIN_MODEL_FEATURES,
    MIN_OOS_PREDICTIONS,
    MIN_STOCK_WEIGHT,
    MIN_TRAIN_MONTHS,
    MIN_VALIDATION_IMPROVEMENT,
    MLP_MIN_TRAIN_MONTHS,
    MLP_PARAMS,
    MLP_REFIT_EVERY,
    NESTED_COMBO_CANDIDATE_POOL,
    OUTER_VALIDATION_MONTHS,
    OPERATIONAL_GATE_PROFILE,
    OPERATIONAL_RISK_ACCEPTANCE_FILE,
    INNER_VALIDATION_MONTHS,
    PERCENTILE_BREAKS,
    PERCENTILE_MIN_HISTORY,
    PERCENTILE_WEIGHTS,
    POSITION_SIZING_POLICIES,
    RANDOM_SEED,
    RAW_TOP_FEATURES_PER_BASE,
    RAW_SERIES_CATALOG_FILE,
    REQUIRED_CORE_FAMILIES,
    REQUIRED_CORE_TRANSFORM_TOKENS,
    RESEARCH_HOLDOUT_START,
    RESEARCH_VALIDATION_MONTHS,
    RESULT_DIR,
    RUNS_DIR,
    SELECTION_MODEL_TYPE,
    SHARPE_ROLLING_WINDOW,
    SMOOTHED_LINEAR_SPAN,
    SINGLE_FACTOR_REFIT_EVERY,
    SINGLE_FACTOR_MODEL_TYPE,
    SVM_CALIBRATION_SPLITS,
    SVM_MIN_TRAIN_MONTHS,
    SVM_PARAMS,
    STATIC_FALLBACK_WEIGHT,
    TARGET_RETURN_THRESHOLD,
    TOP_FEATURE_POOL,
    TRANSACTION_COST_BPS,
    TRANSACTION_COST_SCENARIOS_BPS,
)
from src.data import (
    build_monthly_panel,
    read_investable_price_csv,
    read_market_daily_csv,
)
from src.experiment import (
    build_manifest,
    create_run_directory,
    sha256_file,
    write_manifest,
)
from src.economic_review import add_economic_review_drafts
from src.market_breadth import load_market_breadth, market_breadth_metadata
from src.marcap_kospi200 import ensure_marcap_kospi200_proxy
from src.features import (
    build_feature_matrix,
    build_cross_asset_features,
    build_market_index_features,
    compact_candidate_columns,
    transformation_family,
)
from src.markets import (
    MARKET_PROFILES,
    get_market_profile,
    marketize_raw_catalog,
    mlp_params_for_market,
    mlp_target_spec_for_market,
    primary_target_spec_for_market,
    optional_market_families,
    predeclared_mlp_feature_provenance,
    predeclared_mlp_features,
    predeclared_primary_feature_provenance,
    predeclared_primary_features,
    required_core_families,
    required_core_transform_tokens,
)
from src.modeling import (
    build_candidate_funnel,
    build_model_target,
    chronological_split,
    earliest_walk_forward_prediction_date,
    evaluate_probabilities,
    exhaustive_combination_selection,
    ews_state,
    fit_latest_ews,
    prune_correlated_features,
    round_robin_group_candidates,
    screen_single_factors,
    select_required_core_features,
    walk_forward_predict,
)
from src.visualize import (
    plot_dashboard,
    plot_latest_allocation,
    plot_model_comparison,
    plot_reliability,
)
from src.position_sizing import target_weight_from_ews
from src.raw_catalog import (
    assert_expanded_universe,
    build_raw_series_coverage,
    load_raw_series_catalog,
)
from src.shadow import canonical_spec_hash, initialize_shadow_ledger
from src.splits import research_view
from src.validation import (
    block_bootstrap_policy,
    calibration_diagnostics,
    coefficient_family_sign_stability,
    coefficient_sign_stability,
    compare_position_sizing,
    enforce_signal_gate_fallback,
    evaluate_holdout_safety_veto,
    evaluate_signal_gate,
    fixed_fold_availability_audit,
    fold_candidate_availability_audit,
    logistic_fold_coefficient_audit,
    make_purged_outer_folds,
    fixed_outer_predict,
    nested_outer_predict,
    rolling_active_diagnostics,
    screening_evaluation_start,
    select_position_policy,
)


MODEL_LABELS = {
    "logistic": "Logistic",
    "spline_logistic": "Spline Logistic",
    "svm": "SVM",
    "svm_rank": "RBF SVM Rank",
    "mlp": "MLP",
}

EVALUATED_MODELS = ("logistic", "svm", "mlp")


def _configure_utf8_console():
    """Keep redirected Windows runs from failing on Korean/emoji diagnostics."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _model_min_train_months(model_name):
    if model_name == "svm":
        return max(MIN_TRAIN_MONTHS, SVM_MIN_TRAIN_MONTHS)
    if model_name == "mlp":
        return max(MIN_TRAIN_MONTHS, MLP_MIN_TRAIN_MONTHS)
    return MIN_TRAIN_MONTHS


def _model_runtime_kwargs(model_name, *, mlp_params=None):
    if model_name == "svm":
        return {
            "svm_params": SVM_PARAMS,
            "calibration_splits": SVM_CALIBRATION_SPLITS,
            "random_state": RANDOM_SEED,
        }
    if model_name == "mlp":
        return {
            "mlp_params": dict(mlp_params or MLP_PARAMS),
            "random_state": RANDOM_SEED,
        }
    return {"random_state": RANDOM_SEED}


def _last_completed_month(index, asof_date=None):
    """Return the latest month-end that is complete as of the run date."""
    asof = pd.Timestamp(asof_date or pd.Timestamp.today()).tz_localize(None)
    values = pd.DatetimeIndex(index).tz_localize(None)
    completed = values[values.to_period("M") < asof.to_period("M")]
    if completed.empty:
        raise ValueError("No completed market month is available")
    return completed.max().to_period("M").to_timestamp("M")


def extend_prediction_tail(
    prediction,
    X,
    y,
    *,
    eval_end,
    min_train,
    purge,
    refit_every,
    model_type,
    model_kwargs=None,
):
    """Add score-only months whose forward classification label is incomplete."""
    observed = prediction.dropna().sort_index()
    if observed.empty:
        raise ValueError("Cannot extend an empty historical prediction")
    eval_end = pd.Timestamp(eval_end).to_period("M").to_timestamp("M")
    tail_start = observed.index.max() + pd.offsets.MonthEnd(1)
    if tail_start > eval_end:
        return observed
    tail = walk_forward_predict(
        X,
        y,
        eval_start=tail_start,
        eval_end=eval_end,
        min_train=min_train,
        purge=purge,
        refit_every=refit_every,
        model_type=model_type,
        **(model_kwargs or {}),
    ).dropna()
    combined = pd.concat([observed, tail]).sort_index()
    return combined.loc[~combined.index.duplicated(keep="last")]


def _selection_model_type(model_name):
    return "svm_rank" if model_name == "svm" else model_name


def backtest_evaluation_window(backtest, evaluation_index=None):
    """Return only months in which the dynamic strategy could trade."""

    required = {
        "stock_weight",
        "market_return",
        "strategy_return",
        "benchmark_50_50_return",
        "cash_return",
    }
    missing = required.difference(backtest.columns)
    if missing:
        raise KeyError(
            "백테스트 필수 컬럼 누락: "
            + ", ".join(sorted(missing))
        )

    valid_mask = (
        backtest["stock_weight"].notna()
        & backtest["market_return"].notna()
    )
    valid_index = backtest.index[valid_mask]

    if evaluation_index is not None:
        evaluation_index = pd.Index(evaluation_index)
        unavailable = evaluation_index.difference(valid_index)
        if len(unavailable):
            raise ValueError(
                "요청한 공통 평가기간 중 거래 불가능한 월이 있어: "
                f"{unavailable[0]}"
            )
        valid_index = evaluation_index

    if len(valid_index) == 0:
        raise ValueError("성과를 평가할 공통 거래기간이 없어.")

    return backtest.loc[valid_index].copy()


def evaluate_backtest(
    backtest,
    name,
    evaluation_index=None,
    allocation_policy=None,
    benchmark_name=None,
):
    """Evaluate dynamic and all benchmarks on one identical window."""

    bt = backtest_evaluation_window(
        backtest,
        evaluation_index=evaluation_index,
    )
    avg_weight = bt["stock_weight"].mean()

    # run_backtest의 same-exposure 값이 더 긴 기간에서 계산됐을 수
    # 있으므로 실제 공통 평가기간의 평균 비중으로 다시 계산한다.
    bt["same_exposure_return"] = (
        avg_weight * bt["market_return"]
        + (1.0 - avg_weight) * bt["cash_return"]
    )

    definitions = [
        (f"{name} Dynamic", "strategy_return"),
        (f"{name} Static 50/50", "benchmark_50_50_return"),
        (
            f"{name} Same Exposure ({avg_weight:.1%})",
            "same_exposure_return",
        ),
        (f"{name} {benchmark_name or MARKET_NAME} 100%", "market_return"),
    ]

    rows = []
    for strategy_name, return_column in definitions:
        stats = performance_stats(
            returns=bt[return_column],
            risk_free=bt["cash_return"],
        )
        if stats.get("Months") != len(bt):
            raise AssertionError(
                f"{strategy_name}의 평가 개월이 공통기간과 달라."
            )
        rows.append(
            {
                "model": name,
                "strategy": strategy_name,
                "allocation_policy": (
                    allocation_policy
                    if return_column == "strategy_return"
                    else "benchmark"
                ),
                "Start": bt.index.min().date().isoformat(),
                "End": bt.index.max().date().isoformat(),
                **stats,
            }
        )

    return rows


def class_balance_stats(prediction, y):
    """Compute the accuracy of an always-majority-class baseline."""

    evaluation = pd.concat(
        [prediction.rename("p"), y.rename("y")],
        axis=1,
        sort=True,
    ).dropna()
    if evaluation.empty:
        raise ValueError("Class balance를 계산할 예측값이 없어.")

    positive_rate = float(evaluation["y"].mean())
    naive_accuracy = max(positive_rate, 1.0 - positive_rate)
    return {
        "class_n": len(evaluation),
        "risk_on_rate": positive_rate,
        "risk_off_rate": 1.0 - positive_rate,
        "majority_class": "Risk-On" if positive_rate >= 0.5 else "Risk-Off",
        "naive_accuracy": naive_accuracy,
    }


def _model_comparison_row(model_name, prediction, y):
    metrics = evaluate_probabilities(prediction, y)
    balance = class_balance_stats(prediction, y)
    accuracy = metrics.get("accuracy", np.nan)
    auc = metrics.get("auc", np.nan)

    return {
        "model": MODEL_LABELS[model_name],
        **metrics,
        **balance,
        "accuracy_lift_vs_naive": (
            accuracy - balance["naive_accuracy"]
            if np.isfinite(accuracy)
            else np.nan
        ),
        # 방향 이상을 진단하기 위한 값일 뿐, 실전 신호를 뒤집지 않는다.
        "inverted_auc_diagnostic": (
            1.0 - auc if np.isfinite(auc) else np.nan
        ),
    }


def _fold_signal_gate(
    folds,
    prediction,
    y,
    future_return,
    *,
    eligibility_starts=None,
):
    """Evaluate one model on every pre-holdout outer fold."""

    eligibility_starts = eligibility_starts or {}
    rows = []
    for fold in folds:
        declared_start = eligibility_starts.get(fold.fold, fold.outer_start)
        declared_start = (
            pd.Timestamp(declared_start)
            if pd.notna(declared_start)
            else fold.outer_start
        )
        evaluation_start = max(fold.outer_start, declared_start)
        fold_prediction = prediction.loc[evaluation_start : fold.outer_end]
        fold_metrics = evaluate_probabilities(fold_prediction, y)
        fold_return_data = pd.concat(
            [
                fold_prediction.rename("signal"),
                future_return.rename("future_return"),
            ],
            axis=1,
            sort=True,
        ).dropna()
        fold_rank_ic = (
            fold_return_data["signal"].corr(
                fold_return_data["future_return"], method="spearman"
            )
            if len(fold_return_data) >= 3
            and fold_return_data["signal"].nunique() > 1
            and fold_return_data["future_return"].nunique() > 1
            else np.nan
        )
        expected_observations = int(
            y.loc[evaluation_start : fold.outer_end].notna().sum()
            if evaluation_start <= fold.outer_end
            else 0
        )
        rows.append(
            {
                "fold": fold.fold,
                "start": fold.outer_start,
                "end": fold.outer_end,
                "model_eligibility_start": declared_start,
                "evaluation_start": evaluation_start,
                "expected_observations": expected_observations,
                "observations": int(fold_metrics["n"]),
                "prediction_coverage": (
                    fold_metrics["n"] / expected_observations
                    if expected_observations
                    else np.nan
                ),
                "auc": fold_metrics["auc"],
                "rank_ic": fold_rank_ic,
            }
        )
    return pd.DataFrame(rows)


def _fold_eligibility_map(selections):
    if selections.empty or "model_eligibility_start" not in selections:
        return {}
    values = selections.dropna(subset=["model_eligibility_start"])
    return (
        values.groupby("fold")["model_eligibility_start"]
        .first()
        .map(pd.Timestamp)
        .to_dict()
    )


def _json_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _manifest_config(
    market_profile,
    investable_market_file=None,
    *,
    portfolio_instrument=None,
    portfolio_ticker=None,
    portfolio_return_role=None,
    core_families=None,
    core_transform_tokens=None,
    optional_families=None,
    mlp_params=None,
    fixed_mlp_features=None,
    mlp_target_spec=None,
    primary_target_spec=None,
    fixed_primary_features=None,
    primary_feature_provenance=None,
):
    return {
        "market_key": market_profile.key,
        "market_name": market_profile.display_name,
        "market_series": market_profile.series_name,
        "market_ticker": market_profile.ticker,
        "signal_market_role": f"{market_profile.display_name} price index target",
        "investable_instrument": market_profile.investable_instrument,
        "investable_ticker": market_profile.investable_ticker,
        "portfolio_benchmark_instrument": (
            portfolio_instrument or market_profile.investable_instrument
        ),
        "portfolio_benchmark_ticker": portfolio_ticker,
        "portfolio_return_role": portfolio_return_role or (
            "audited investable return series"
            if investable_market_file is not None
            else f"{market_profile.display_name} price-index proxy; operational gate blocked"
        ),
        "forecast_horizon_months": FORECAST_HORIZON,
        "target_return_threshold": TARGET_RETURN_THRESHOLD,
        "research_holdout_start": RESEARCH_HOLDOUT_START,
        "research_validation_months": RESEARCH_VALIDATION_MONTHS,
        "min_train_months": MIN_TRAIN_MONTHS,
        "svm_min_train_months": SVM_MIN_TRAIN_MONTHS,
        "outer_validation_months": OUTER_VALIDATION_MONTHS,
        "inner_validation_months": INNER_VALIDATION_MONTHS,
        "single_factor_model": SINGLE_FACTOR_MODEL_TYPE,
        "selection_model": SELECTION_MODEL_TYPE,
        "final_model": FINAL_MODEL_TYPE,
        "position_sizing_policies": list(POSITION_SIZING_POLICIES),
        "smoothed_linear_span_months": SMOOTHED_LINEAR_SPAN,
        "static_fallback_stock_weight": STATIC_FALLBACK_WEIGHT,
        "allocation_fail_closed_rule": (
            "use static strategic allocation unless both the pre-holdout signal "
            "gate and a tactical portfolio-policy gate pass"
        ),
        "transaction_cost_scenarios_bps": list(TRANSACTION_COST_SCENARIOS_BPS),
        "raw_series_catalog": str(RAW_SERIES_CATALOG_FILE),
        "raw_series_target": 70,
        "minimum_expanded_raw_series": MIN_EXPANDED_RAW_SERIES,
        "top_single_factor_pool": TOP_FEATURE_POOL,
        "raw_top_features_per_base": RAW_TOP_FEATURES_PER_BASE,
        "group_candidates_per_group": GROUP_CANDIDATES_PER_GROUP,
        "max_features_per_base": MAX_FEATURES_PER_BASE,
        "max_features_per_group": MAX_FEATURES_PER_GROUP,
        "minimum_distinct_groups": MIN_DISTINCT_GROUPS,
        "required_core_families": core_families or REQUIRED_CORE_FAMILIES,
        "required_core_transform_tokens": (
            core_transform_tokens or REQUIRED_CORE_TRANSFORM_TOKENS
        ),
        "optional_market_families": optional_families or {},
        "minimum_model_features": MIN_MODEL_FEATURES,
        "maximum_model_features": MAX_MODEL_FEATURES,
        "evaluated_models": list(EVALUATED_MODELS),
        "mlp_params": dict(mlp_params or MLP_PARAMS),
        "predeclared_mlp_features": list(fixed_mlp_features or []),
        "mlp_target_spec": dict(mlp_target_spec or {}),
        "primary_target_spec": dict(primary_target_spec or {}),
        "predeclared_primary_features": list(fixed_primary_features or []),
        "primary_feature_provenance": primary_feature_provenance,
        "mlp_min_train_months": MLP_MIN_TRAIN_MONTHS,
        "mlp_refit_every_months": MLP_REFIT_EVERY,
        "random_seed": RANDOM_SEED,
        "spec_source": "PLAN.md and supplied original EWS technical slides",
        "cash_return_convention": "prior month annualized DGS3MO percent divided by 12",
        "point_in_time_rule": (
            "selected FRED sources require hash-validated month-end ALFRED vintages; "
            "derived market observations execute next month"
        ),
        "operational_gate_profile": OPERATIONAL_GATE_PROFILE,
        "hp_filter": "excluded; no two-sided future-dependent transforms",
        "tracks": {
            "original_reference": "exact candidates + fast expanding univariate logistic + Logistic combination + linear sizing baseline",
            "robust_research": "raw top-3 + group top-15 + group-diverse Logistic champion + nested purged validation",
            "challengers": "SVM and MLP choose their own pre-holdout feature combinations; historical holdout diagnostics only",
            "external_reference": (
                "late-2025 through 2026-04 screenshot variables; diagnostic only, "
                "never used to choose or tune the research model"
            ),
        },
    }


def build_external_reference_coverage(registry, factor_candidates, X):
    """Resolve screenshot variables without feeding holdout observations to selection."""

    candidate_lookup = factor_candidates.set_index("feature")
    rows = []
    for _, reference in registry.iterrows():
        mapped = str(reference["mapped_feature"])
        if mapped in X.columns:
            resolved = mapped
            transform_status = "direct_curated_feature"
        elif f"{mapped}__level" in X.columns:
            resolved = f"{mapped}__level"
            transform_status = "base_source_level_only"
        else:
            resolved = None
            transform_status = "unavailable"

        candidate = (
            candidate_lookup.loc[resolved]
            if resolved is not None and resolved in candidate_lookup.index
            else None
        )
        series = X[resolved].dropna() if resolved is not None else pd.Series(dtype=float)
        rows.append(
            {
                **reference.to_dict(),
                "resolved_feature": resolved,
                "transform_mapping_status": transform_status,
                "available_in_factor_matrix": resolved is not None,
                "first_available": (
                    series.index.min().date().isoformat() if not series.empty else None
                ),
                "last_available": (
                    series.index.max().date().isoformat() if not series.empty else None
                ),
                "observations": int(series.size),
                "eligible_original_track": (
                    bool(candidate["eligible_original_track"])
                    if candidate is not None
                    else False
                ),
                "eligible_robust_track": (
                    bool(candidate["eligible_robust_track"])
                    if candidate is not None
                    else False
                ),
                "holdout_selection_prohibited": True,
            }
        )
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the EWS research pipeline")
    parser.add_argument("--run-id", help="immutable run directory name")
    parser.add_argument(
        "--market",
        choices=MARKET_PROFILES,
        default="kospi200",
        help="prediction target and investable benchmark market",
    )
    parser.add_argument(
        "--market-file",
        help="optional override for the selected signal-index OHLCV CSV",
    )
    parser.add_argument(
        "--market-metadata-file",
        help="optional override for the signal-index metadata JSON",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="skip repeated nested feature search; for smoke testing only",
    )
    parser.add_argument(
        "--allow-partial-raw-universe",
        action="store_true",
        help="permit fewer than 50 available raw sensors for smoke tests only",
    )
    parser.add_argument(
        "--investable-market-file",
        help=(
            "audited total-return index or ETF adjusted-price CSV; requires a "
            "date column and adjusted_close/adj_close/Adj Close/total_return_index"
        ),
    )
    return parser.parse_args()


def main(
    run_id=None,
    quick=False,
    allow_partial_raw_universe=False,
    market_key="kospi200",
    market_file=None,
    market_metadata_file=None,
    investable_market_file=None,
):
    _configure_utf8_console()
    global RESULT_DIR, MARKET_NAME, MARKET_SERIES_NAME
    root = Path(__file__).resolve().parent
    market_profile = get_market_profile(market_key)
    active_mlp_params = mlp_params_for_market(market_profile, MLP_PARAMS)
    active_mlp_target_spec = mlp_target_spec_for_market(market_profile)
    active_primary_target_spec = primary_target_spec_for_market(market_profile)
    fixed_primary_features = predeclared_primary_features(market_profile)
    primary_feature_provenance = predeclared_primary_feature_provenance(
        market_profile
    )
    fixed_mlp_features = predeclared_mlp_features(market_profile)
    fixed_mlp_feature_provenance = predeclared_mlp_feature_provenance(
        market_profile
    )
    MARKET_NAME = market_profile.display_name
    MARKET_SERIES_NAME = market_profile.series_name
    market_file = Path(market_file or market_profile.signal_file)
    market_metadata_file = Path(
        market_metadata_file
        or (
            Path(market_file).with_suffix(".metadata.json")
            if market_file != market_profile.signal_file
            else market_profile.signal_metadata_file
        )
    )
    core_families = required_core_families(market_profile)
    core_transform_tokens = required_core_transform_tokens(market_profile)
    optional_families = optional_market_families(market_profile)
    explicit_investable_override = investable_market_file is not None
    use_marcap_kospi200_proxy = (
        market_profile.key == "kospi200" and not explicit_investable_override
    )
    if use_marcap_kospi200_proxy:
        ensure_marcap_kospi200_proxy(
            KOREA_STOCK_PANEL_FILE,
            MARCAP_KOSPI200_FILE,
            MARCAP_KOSPI200_METADATA_FILE,
        )
        investable_market_file = Path(MARCAP_KOSPI200_FILE)
        portfolio_instrument = "Marcap KOSPI Top-200 Price Proxy"
        portfolio_ticker = None
        portfolio_return_role = (
            "point-in-time monthly-rebalanced KOSPI market-cap top-200 price proxy"
        )
    else:
        investable_market_file = (
            Path(investable_market_file)
            if investable_market_file is not None
            else market_profile.investable_file
        )
        portfolio_instrument = market_profile.investable_instrument
        portfolio_ticker = market_profile.investable_ticker
        portfolio_return_role = "audited investable return series"
    if not market_file.is_file():
        raise FileNotFoundError(
            f"Missing {market_profile.display_name} market file: {market_file}. "
            f"Run: python download_market_data.py --market {market_profile.key}"
        )
    if not investable_market_file.is_file():
        raise FileNotFoundError(
            f"Missing {market_profile.investable_instrument} file: "
            f"{investable_market_file}. Run: python download_market_data.py "
            f"--market {market_profile.key}"
        )
    RESULT_DIR = create_run_directory(RUNS_DIR, run_id=run_id)
    data_files = (
        list(Path(FRED_DIR).glob("*.csv"))
        + list(Path(ALFRED_DIR).glob("*.csv"))
        + list(Path(ALFRED_DIR).glob("*.metadata.json"))
        + [
        Path(RAW_SERIES_CATALOG_FILE),
        Path(EXACT_INDICATOR_GAP_FILE),
        market_file,
        market_metadata_file,
        Path(ECONOMIC_REVIEW_FILE),
        Path(EXTERNAL_REFERENCE_FILE),
        Path(MARKET_BREADTH_FILE),
        Path(MARKET_BREADTH_METADATA_FILE),
        Path(OPERATIONAL_RISK_ACCEPTANCE_FILE),
        ]
    )
    if investable_market_file is not None:
        data_files.append(Path(investable_market_file))
        investable_metadata = Path(investable_market_file).with_suffix(
            ".metadata.json"
        )
        if investable_metadata.is_file():
            data_files.append(investable_metadata)
    code_files = [
        root / "run_pipeline.py",
        root / "run_mlp_preholdout_research.py",
        root / "approve_economic_review.py",
        root / "score_frozen_mlp.py",
        root / "shadow_monitor.py",
        root / "requirements.txt",
    ]
    code_files.extend((root / "src").glob("*.py"))
    manifest = build_manifest(
        root=root,
        output_dir=RESULT_DIR,
        config=_manifest_config(
            market_profile,
            investable_market_file,
            core_families=core_families,
            core_transform_tokens=core_transform_tokens,
            optional_families=optional_families,
            mlp_params=active_mlp_params,
            fixed_mlp_features=fixed_mlp_features,
            mlp_target_spec=active_mlp_target_spec,
            primary_target_spec=active_primary_target_spec,
            fixed_primary_features=fixed_primary_features,
            primary_feature_provenance=primary_feature_provenance,
            portfolio_instrument=portfolio_instrument,
            portfolio_ticker=portfolio_ticker,
            portfolio_return_role=portfolio_return_role,
        ),
        data_files=data_files,
        code_files=code_files,
        status="running",
        extra={
            "quick_smoke_test": bool(quick),
            "partial_raw_universe_allowed": bool(allow_partial_raw_universe),
        },
    )
    write_manifest(RESULT_DIR, manifest)

    print()
    print("=" * 70)
    print("[START] EWS MACRO MODEL START")
    print("=" * 70)
    print(f"Run directory: {RESULT_DIR}")
    print(
        f"Market: {market_profile.display_name} ({market_profile.ticker}) | "
        f"portfolio benchmark: {portfolio_instrument}"
    )

    # ① DATA
    print()
    print("① FRED 데이터 월간 통합")
    raw_catalog = marketize_raw_catalog(
        load_raw_series_catalog(RAW_SERIES_CATALOG_FILE), market_profile
    )
    exact_indicator_gaps = pd.read_csv(EXACT_INDICATOR_GAP_FILE)
    if not exact_indicator_gaps["selection_use"].eq(
        "prohibited_until_exact_source"
    ).all():
        raise ValueError("Exact-source gaps must be prohibited from model selection")
    exact_indicator_gaps.to_csv(
        RESULT_DIR / "exact_indicator_gap_audit.csv", index=False
    )
    panel, metadata = build_monthly_panel(FRED_DIR, RAW_SERIES_CATALOG_FILE)
    fred_coverage = build_raw_series_coverage(raw_catalog, FRED_DIR)
    eligible_fred_names = set(
        fred_coverage.loc[
            fred_coverage["source"].eq("FRED")
            & fred_coverage["research_eligible"],
            "name",
        ]
    )
    panel = panel[[name for name in panel.columns if name in eligible_fred_names]]
    metadata = metadata.loc[metadata["name"].isin(eligible_fred_names)].copy()
    panel.to_parquet(RESULT_DIR / "monthly_panel.parquet")
    metadata.to_csv(RESULT_DIR / "series_metadata.csv", index=False)
    point_in_time = point_in_time_audit(metadata)
    point_in_time.to_csv(RESULT_DIR / "point_in_time_audit.csv", index=False)
    print(f"원자료 개수: {panel.shape[1]:,}")
    print(f"월간 행 개수: {panel.shape[0]:,}")

    market_daily = read_market_daily_csv(market_file)
    market = market_daily["close"].resample("ME").last()
    market.name = MARKET_SERIES_NAME
    market.to_csv(RESULT_DIR / "market_monthly.csv")
    if market_profile.key == "kospi200":
        market.to_csv(RESULT_DIR / "kospi200_monthly.csv")
    investable_distribution_adjusted = False
    if investable_market_file is not None:
        investable_price = read_investable_price_csv(investable_market_file)
        investable_distribution_adjusted = bool(
            investable_price.attrs.get("distribution_adjusted", False)
        )
        portfolio_market = investable_price.resample("ME").last()
        portfolio_market.name = f"investable_{MARKET_SERIES_NAME}"
    else:
        portfolio_market = market.copy()
        portfolio_market.name = f"{MARKET_SERIES_NAME}_price_index_portfolio_proxy"
    portfolio_market.to_csv(RESULT_DIR / "portfolio_return_source_monthly.csv")
    market_roles = market_return_role_registry(
        market_file,
        investable_market_file,
        investable_distribution_adjusted=investable_distribution_adjusted,
        market_name=market_profile.display_name,
        market_ticker=market_profile.ticker,
        investable_instrument=portfolio_instrument,
        investable_ticker=portfolio_ticker,
        portfolio_return_type=(
            "point-in-time market-cap-weighted price return proxy"
            if use_marcap_kospi200_proxy
            else None
        ),
        portfolio_notes=(
            "marcap KOSPI common-share top-200 proxy; official historical "
            "KOSPI200 membership is unavailable; excludes distributions"
            if use_marcap_kospi200_proxy
            else None
        ),
        portfolio_deployment_eligible=(
            False if use_marcap_kospi200_proxy else None
        ),
    )
    market_roles.to_csv(RESULT_DIR / "market_return_role_registry.csv", index=False)

    # ② FACTOR FACTORY
    print()
    print("② Factor Factory")
    macro_X = build_feature_matrix(panel, metadata)
    market_factors, market_factor_metadata = (
        build_market_index_features(
            market_daily,
            series_name=market_profile.series_name,
            market_name=market_profile.display_name,
            ticker=market_profile.ticker,
        )
    )
    breadth_factors, breadth_cache_metadata = load_market_breadth(
        KOREA_STOCK_PANEL_FILE,
        MARKET_BREADTH_FILE,
        MARKET_BREADTH_METADATA_FILE,
    )
    breadth_factor_metadata = market_breadth_metadata(breadth_factors)
    breadth_eligible = breadth_factor_metadata.loc[
        breadth_factor_metadata["model_eligible"], "feature"
    ].tolist()
    market_factors.to_csv(
        RESULT_DIR / "market_index_factors.csv"
    )
    market_factor_metadata.to_csv(
        RESULT_DIR / "market_index_factor_metadata.csv",
        index=False,
    )
    breadth_factors.to_parquet(RESULT_DIR / "korea_market_breadth.parquet")
    breadth_factor_metadata.to_csv(
        RESULT_DIR / "korea_market_breadth_metadata.csv", index=False
    )
    (RESULT_DIR / "korea_market_breadth_cache_metadata.json").write_text(
        json.dumps(breadth_cache_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    freshness_audit = data_freshness_audit(
        metadata,
        market_daily,
        breadth_cache_metadata,
        market_series_name=market_profile.series_name,
    )
    freshness_audit.to_csv(RESULT_DIR / "data_freshness_schema_audit.csv", index=False)

    cross_asset_factors, cross_asset_metadata = build_cross_asset_features(panel)
    derived_sources = pd.concat(
        [
            market_factors,
            breadth_factors[breadth_eligible],
            cross_asset_factors,
        ],
        axis=1,
        sort=True,
    )
    derived_catalog = raw_catalog.loc[
        raw_catalog["enabled"] & raw_catalog["source"].str.startswith("derived_")
    ].copy()
    missing_derived = sorted(set(derived_catalog["name"]) - set(derived_sources.columns))
    if missing_derived:
        raise KeyError(f"Catalogued derived raw series are unavailable: {missing_derived}")
    derived_panel = derived_sources[derived_catalog["name"].tolist()]
    source_metadata = pd.concat(
        [
            market_factor_metadata,
            breadth_factor_metadata,
            cross_asset_metadata,
        ],
        ignore_index=True,
    ).drop_duplicates("feature").set_index("feature")
    derived_metadata = derived_catalog.rename(columns={"file": "catalog_file"}).copy()
    derived_metadata["file"] = derived_metadata["name"].map(
        lambda name: str(source_metadata.loc[name, "source"])
    )
    derived_metadata["source"] = derived_metadata["file"]
    derived_metadata["exactness"] = derived_metadata["name"].map(
        source_metadata["exactness"]
    )
    derived_metadata["first_date"] = derived_metadata["name"].map(
        lambda name: derived_panel[name].first_valid_index()
    )
    derived_metadata["last_date"] = derived_metadata["name"].map(
        lambda name: derived_panel[name].last_valid_index()
    )
    derived_X = build_feature_matrix(derived_panel, derived_metadata)

    raw_coverage = build_raw_series_coverage(
        raw_catalog,
        FRED_DIR,
        available_derived_names=derived_sources,
    )
    raw_coverage.to_csv(RESULT_DIR / "raw_series_coverage.csv", index=False)
    raw_group_summary = (
        raw_coverage.groupby("group", sort=False)
        .agg(
            configured=("name", "size"),
            available=("available", "sum"),
            research_eligible=("research_eligible", "sum"),
        )
        .reset_index()
    )
    raw_group_summary.to_csv(
        RESULT_DIR / "raw_series_group_summary.csv", index=False
    )
    if not allow_partial_raw_universe:
        assert_expanded_universe(raw_coverage, MIN_EXPANDED_RAW_SERIES)
    print(
        "Raw universe: "
        f"catalog {len(raw_coverage):,}, "
        f"available {int(raw_coverage['available'].sum()):,}, "
        f"research-eligible {int(raw_coverage['research_eligible'].sum()):,}"
    )
    print(raw_group_summary.to_string(index=False))

    X = pd.concat(
        [macro_X, derived_X, market_factors, breadth_factors[breadth_eligible]],
        axis=1,
    ).sort_index()
    X.index.name = "observation_date"
    X.to_parquet(RESULT_DIR / "factor_matrix.parquet")
    pd.DataFrame(
        [
            {"transform_family": "level/change/MA/EWMA/z/vol/slope", "included": True,
             "causal_rule": "rolling/expanding calculations use observations through t only",
             "truncate_test_required": True},
            {"transform_family": "two-sided_HP_filter", "included": False,
             "causal_rule": "excluded because a full-sample two-sided filter leaks future values",
             "truncate_test_required": False},
        ]
    ).to_csv(RESULT_DIR / "transformation_causality_registry.csv", index=False)
    print(
        f"[FACTORS] 생성 Factor 수: {X.shape[1]:,} "
        f"(FRED Factory {macro_X.shape[1]:,} + "
        f"derived raw Factory {derived_X.shape[1]:,} + "
        f"{MARKET_NAME} curated {market_factors.shape[1]:,} + "
        f"Korea stock-universe breadth {len(breadth_eligible):,})"
    )

    factory_metadata = pd.concat(
        [metadata, derived_metadata], ignore_index=True, sort=False
    )
    factory_X = pd.concat([macro_X, derived_X], axis=1)
    macro_group_lookup = (
        factory_metadata.drop_duplicates("name")
        .set_index("name")["group"]
        .to_dict()
    )
    macro_lag_lookup = (
        factory_metadata.drop_duplicates("name")
        .set_index("name")["availability_lag"]
        .to_dict()
    )
    macro_file_lookup = (
        factory_metadata.drop_duplicates("name")
        .set_index("name")["file"]
        .to_dict()
    )
    declared_proxy_bases = set(
        factory_metadata.loc[
            factory_metadata["name"].str.contains("proxy", case=False, na=False)
            | factory_metadata["sensor_family"].str.contains(
                "proxy", case=False, na=False
            ),
            "name",
        ]
    )
    macro_candidates = pd.DataFrame(
        {
            "feature": factory_X.columns,
            "base": [
                feature.split("__")[0]
                for feature in factory_X.columns
            ],
            "source": "Expanded Raw-Series Factor Factory",
        }
    )
    macro_candidates["group"] = macro_candidates["base"].map(
        macro_group_lookup
    ).fillna("unknown")
    macro_candidates["source_file"] = macro_candidates["base"].map(
        macro_file_lookup
    )
    macro_candidates["availability_lag"] = macro_candidates["base"].map(
        macro_lag_lookup
    )
    macro_candidates["family"] = macro_candidates["feature"].map(
        transformation_family
    )
    derived_exactness = derived_metadata.set_index("name")["exactness"].to_dict()
    macro_candidates["exactness"] = macro_candidates["base"].map(
        lambda base: (
            f"derived_from_{derived_exactness[base]}"
            if base in derived_exactness
            else "derived_from_declared_proxy"
            if base in declared_proxy_bases
            else "derived_from_exact_source"
        )
    )
    macro_candidates["point_in_time_rule"] = (
        "configured publication lag; historical revisions not vintage-safe"
    )
    macro_candidates["description"] = (
        "FRED 원자료의 Factor Factory 변환"
    )
    market_factor_metadata = market_factor_metadata.copy()
    market_factor_metadata["group"] = "market_internal"
    breadth_factor_metadata = breadth_factor_metadata.copy()
    breadth_factor_metadata["group"] = "kospi_internal"
    market_factor_metadata["model_eligible"] = True
    factor_candidates = pd.concat(
        [
            macro_candidates,
            market_factor_metadata[
                [
                    "feature",
                    "base",
                    "source",
                    "group",
                    "exactness",
                    "description",
                    "availability_lag",
                    "point_in_time_rule",
                    "model_eligible",
                ]
            ],
            breadth_factor_metadata[
                [
                    "feature",
                    "base",
                    "source",
                    "group",
                    "exactness",
                    "description",
                    "availability_lag",
                    "point_in_time_rule",
                    "model_eligible",
                ]
            ],
        ],
        ignore_index=True,
    )
    factor_candidates["family"] = factor_candidates["feature"].map(
        transformation_family
    )
    factor_candidates["model_eligible"] = factor_candidates[
        "model_eligible"
    ].fillna(True).astype(bool)
    factor_candidates["is_proxy"] = factor_candidates["exactness"].str.contains(
        "proxy", na=False
    )
    factor_candidates["eligible_original_track"] = (
        factor_candidates["model_eligible"]
        & ~factor_candidates["is_proxy"]
        & ~factor_candidates["exactness"].str.contains("direct_universe", na=False)
    )
    factor_candidates["eligible_robust_track"] = (
        factor_candidates["model_eligible"] & ~factor_candidates["is_proxy"]
    )
    factor_candidates["compact_nested_search"] = factor_candidates["feature"].isin(
        compact_candidate_columns(X.columns)
    )
    candidate_operational_audit = selected_point_in_time_audit(
        factor_candidates["feature"].tolist(),
        raw_catalog,
        point_in_time,
        feature_metadata=factor_candidates,
    ).drop_duplicates("feature")
    candidate_operational_audit.to_csv(
        RESULT_DIR / "candidate_point_in_time_eligibility.csv", index=False
    )
    candidate_operational_lookup = candidate_operational_audit.set_index("feature")
    factor_candidates["strict_vintage_gate_passed"] = factor_candidates[
        "feature"
    ].map(candidate_operational_lookup["strict_vintage_gate_passed"]).fillna(False)
    factor_candidates["release_timing_gate_passed"] = factor_candidates[
        "feature"
    ].map(candidate_operational_lookup["release_timing_gate_passed"]).fillna(False)
    factor_candidates["eligible_mlp_deployment_track"] = (
        factor_candidates["eligible_robust_track"]
        & factor_candidates["strict_vintage_gate_passed"]
        & factor_candidates["release_timing_gate_passed"]
    )
    factor_candidates.to_csv(
        RESULT_DIR / "factor_candidates.csv",
        index=False,
    )
    unknown_groups = sorted(
        factor_candidates.loc[
            factor_candidates["group"].eq("unknown"), "base"
        ].dropna().unique()
    )
    if unknown_groups:
        raise ValueError(f"Unregistered economic groups: {unknown_groups}")

    reference_registry = pd.read_csv(EXTERNAL_REFERENCE_FILE)
    if not reference_registry["selection_use"].eq("diagnostic_only").all():
        raise ValueError("External holdout references must remain diagnostic_only")
    reference_coverage = build_external_reference_coverage(
        reference_registry, factor_candidates, X
    )
    reference_coverage.to_csv(
        RESULT_DIR / "external_reference_coverage.csv", index=False
    )
    reference_features = reference_coverage.loc[
        reference_coverage["available_in_factor_matrix"], "resolved_feature"
    ].dropna().unique().tolist()
    reference_values = X[reference_features].loc["2025-12-31":"2026-04-30"]
    reference_values = reference_values.stack(future_stack=True).rename("value").reset_index()
    reference_values = reference_values.rename(columns={"level_1": "resolved_feature"})
    reference_values["selection_use"] = "diagnostic_only"
    reference_values["used_for_tuning"] = False
    reference_values.to_csv(
        RESULT_DIR / "external_reference_observed_window_values.csv", index=False
    )

    target = build_model_target(
        market,
        mode=active_primary_target_spec["mode"],
        horizon=FORECAST_HORIZON,
        return_threshold=active_primary_target_spec.get(
            "return_threshold", TARGET_RETURN_THRESHOLD
        ),
        cash_yield=panel.get("cash_yield_3m"),
    )
    target.to_csv(RESULT_DIR / "target.csv")
    y = target["y"]
    mlp_target = build_model_target(
        market,
        mode=active_mlp_target_spec["mode"],
        horizon=FORECAST_HORIZON,
        return_threshold=active_mlp_target_spec.get(
            "return_threshold", TARGET_RETURN_THRESHOLD
        ),
        drawdown_threshold=active_mlp_target_spec.get(
            "drawdown_threshold", -0.05
        ),
        cash_yield=panel.get("cash_yield_3m"),
    )
    mlp_target.to_csv(RESULT_DIR / "mlp_target.csv")
    mlp_y = mlp_target["y"]

    ineligible_features = set(
        factor_candidates.loc[
            ~factor_candidates["eligible_robust_track"], "feature"
        ]
    )
    compact_features = [
        feature
        for feature in compact_candidate_columns(X.columns)
        if feature not in ineligible_features
    ]
    robust_eligible = set(
        factor_candidates.loc[
            factor_candidates["eligible_robust_track"], "feature"
        ]
    )
    full_robust_features = [feature for feature in X.columns if feature in robust_eligible]
    selection_X = X[full_robust_features]
    nested_selection_X = X[compact_features]
    mlp_deployment_eligible = set(
        factor_candidates.loc[
            factor_candidates["eligible_mlp_deployment_track"], "feature"
        ]
    )
    mlp_compact_features = [
        feature for feature in compact_features if feature in mlp_deployment_eligible
    ]
    mlp_nested_research_X = X[mlp_compact_features]
    pd.Series(mlp_compact_features, name="feature").to_csv(
        RESULT_DIR / "mlp_deployment_candidate_universe.csv", index=False
    )
    pd.Series(full_robust_features, name="feature").to_csv(
        RESULT_DIR / "full_robust_candidate_universe.csv", index=False
    )
    pd.Series(compact_features, name="feature").to_csv(
        RESULT_DIR / "compact_robust_candidate_universe.csv",
        index=False,
    )
    original_eligible = set(
        factor_candidates.loc[
            factor_candidates["eligible_original_track"], "feature"
        ]
    )
    original_features = [feature for feature in X.columns if feature in original_eligible]
    selection_X_original = X[original_features]
    pd.Series(original_features, name="feature").to_csv(
        RESULT_DIR / "original_reference_candidate_universe.csv", index=False
    )
    pd.DataFrame(
        [
            {"stage": "configured_raw_series", "count": len(raw_coverage)},
            {
                "stage": "available_raw_series",
                "count": int(raw_coverage["available"].sum()),
            },
            {
                "stage": "research_eligible_raw_series",
                "count": int(raw_coverage["research_eligible"].sum()),
            },
            {"stage": "factor_factory_output", "count": factory_X.shape[1]},
            {"stage": "all_factor_matrix", "count": X.shape[1]},
            {"stage": "robust_univariate_screen", "count": len(full_robust_features)},
            {"stage": "top_ranked_pool_limit", "count": TOP_FEATURE_POOL},
            {"stage": "correlation_pruned_limit", "count": COMBO_CANDIDATE_POOL},
            {"stage": "bounded_exhaustive_limit", "count": EXHAUSTIVE_COMBO_CANDIDATE_POOL},
            {"stage": "final_model_feature_minimum", "count": MIN_MODEL_FEATURES},
            {"stage": "final_model_feature_limit", "count": MAX_MODEL_FEATURES},
        ]
    ).to_csv(RESULT_DIR / "factor_universe_summary.csv", index=False)

    split = chronological_split(
        y,
        test_start=RESEARCH_HOLDOUT_START,
        validation_months=RESEARCH_VALIDATION_MONTHS,
    )
    target_index = y.dropna().index
    print()
    print("[SPLIT] 시계열 데이터 분리")
    print(
        "development          : "
        f"{target_index.min().date()} ~ {split['dev_end'].date()}"
    )
    print(
        "validation           : "
        f"{split['validation_start'].date()} "
        f"~ {split['validation_end'].date()}"
    )
    print(
        "research holdout     : "
        f"{split['test_start'].date()} ~ {split['test_end'].date()}"
    )
    configured_holdout = pd.Timestamp(RESEARCH_HOLDOUT_START)
    if split["test_start"] != configured_holdout:
        raise AssertionError(
            "Chronological split no longer matches the pre-declared research holdout: "
            f"{split['test_start'].date()} != {configured_holdout.date()}"
        )
    pre_holdout_end = (
        configured_holdout.to_period("M") - 1
    ).to_timestamp("M")
    research_X, research_y = research_view(selection_X, y, pre_holdout_end)
    _, mlp_research_y = research_view(selection_X, mlp_y, pre_holdout_end)
    research_future_return = target["future_return"].loc[:pre_holdout_end].copy()
    pre_holdout_label_end = (
        configured_holdout.to_period("M") - 1 - FORECAST_HORIZON
    ).to_timestamp("M")
    research_y.loc[research_y.index > pre_holdout_label_end] = np.nan
    mlp_research_y.loc[mlp_research_y.index > pre_holdout_label_end] = np.nan
    research_future_return.loc[
        research_future_return.index > pre_holdout_label_end
    ] = np.nan
    nested_research_X, _ = research_view(
        nested_selection_X, y, pre_holdout_end
    )
    original_research_X, _ = research_view(
        selection_X_original, y, pre_holdout_end
    )
    first_oos_position = min(MIN_TRAIN_MONTHS, len(target_index) - 1)
    print(
        "최초 rolling 예측 가능: "
        f"{target_index[first_oos_position].date()} "
        f"(초기 학습 {MIN_TRAIN_MONTHS}개월 필요)"
    )

    # ③ SINGLE FACTOR SCREEN
    print()
    print("③ 단일 Factor ML")
    single_scores = screen_single_factors(
        X=research_X,
        y=research_y,
        dev_end=split["dev_end"],
        eval_start=(split["dev_end"].to_period("M") - 59).to_timestamp("M"),
        min_train=MIN_TRAIN_MONTHS,
        horizon=FORECAST_HORIZON,
        refit_every=SINGLE_FACTOR_REFIT_EVERY,
        n_jobs=-1,
        model_type=SINGLE_FACTOR_MODEL_TYPE,
        svm_params=SVM_PARAMS,
        calibration_splits=SVM_CALIBRATION_SPLITS,
    )
    single_scores.to_csv(
        RESULT_DIR / "single_factor_scores.csv",
        index=False,
    )
    # The primary 10k-scale screen is already an expanding-window univariate
    # logistic model.  Refit-free reuse keeps this compact diagnostic exactly
    # consistent with the selection scores and avoids thousands of duplicate
    # sklearn fits.
    linear_single_scores = single_scores.loc[
        single_scores["feature"].isin(nested_research_X.columns)
    ].copy()
    linear_single_scores["diagnostic_model"] = "fast_logistic_same_primary"
    linear_single_scores["selection_use"] = "interpretation_only"
    linear_single_scores.to_csv(
        RESULT_DIR / "single_factor_logistic_diagnostic.csv", index=False
    )

    usable = single_scores.loc[
        (single_scores["n"] >= MIN_OOS_PREDICTIONS)
        & single_scores["auc"].notna()
    ].sort_values("rank_score", ascending=False)

    print()
    print("🏆 단일 Factor TOP 20")
    print(
        usable[
            [
                "feature",
                "auc",
                "auc_first",
                "auc_second",
                "brier",
                "rank_score",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    top_ranked = usable["feature"].head(TOP_FEATURE_POOL).tolist()
    pd.Series(top_ranked, name="feature").to_csv(
        RESULT_DIR / "top_ranked_candidates.csv", index=False
    )

    feature_groups = (
        factory_metadata.drop_duplicates("name")
        .set_index("name")["group"]
        .to_dict()
    )
    feature_groups.update(
        market_factor_metadata.set_index("base")["group"].to_dict()
    )
    feature_groups.update(
        breadth_factor_metadata.set_index("base")["group"].to_dict()
    )
    required_core_bases = {
        base
        for bases in (*core_families.values(), *optional_families.values())
        for base in bases
    }
    required_core_columns = [
        feature
        for feature in research_X.columns
        if feature.split("__", 1)[0] in required_core_bases
    ]
    if not required_core_columns:
        raise RuntimeError("Required EWS core candidate universe is empty")
    raw_stage, group_stage = build_candidate_funnel(
        usable,
        feature_groups,
        max_per_base=RAW_TOP_FEATURES_PER_BASE,
        max_per_group=GROUP_CANDIDATES_PER_GROUP,
    )
    raw_stage.to_csv(RESULT_DIR / "raw_series_funnel_candidates.csv", index=False)
    group_stage.to_csv(RESULT_DIR / "group_balanced_candidates.csv", index=False)
    ranked = group_stage["feature"].tolist()
    candidates = prune_correlated_features(
        X=research_X,
        ranked_features=ranked,
        dev_end=split["dev_end"],
        threshold=CORRELATION_THRESHOLD,
        max_features=COMBO_CANDIDATE_POOL,
        min_obs=MIN_OOS_PREDICTIONS,
    )
    pd.Series(candidates, name="feature").to_csv(
        RESULT_DIR / "combination_candidates.csv",
        index=False,
    )
    print(f"중복 제거 후 조합 후보: {len(candidates)}개")

    # ④ GROUP-AWARE COMBINATION SEARCH
    print()
    print("④ Factor 조합 탐색 (원천·그룹 hard cap 없음·상관 중복 제거)")
    exhaustive_candidates = round_robin_group_candidates(
        candidates,
        feature_groups,
        max_features=EXHAUSTIVE_COMBO_CANDIDATE_POOL,
    )
    if len(exhaustive_candidates) < MIN_MODEL_FEATURES:
        raise RuntimeError(
            "Not enough candidates survived screening for the predeclared "
            f"{MIN_MODEL_FEATURES}-feature minimum"
        )
    pd.Series(exhaustive_candidates, name="feature").to_csv(
        RESULT_DIR / "bounded_exhaustive_candidates.csv", index=False
    )
    unconstrained_selected_features, search_history = exhaustive_combination_selection(
        X=research_X,
        y=research_y,
        candidates=exhaustive_candidates,
        validation_start=split["validation_start"],
        validation_end=split["validation_end"],
        min_train=MIN_TRAIN_MONTHS,
        horizon=FORECAST_HORIZON,
        min_features=MIN_MODEL_FEATURES,
        max_features=EXHAUSTIVE_MAX_COMBO_SIZE,
        max_features_per_base=MAX_FEATURES_PER_BASE,
        feature_groups=feature_groups,
        max_features_per_group=MAX_FEATURES_PER_GROUP,
        min_distinct_groups=MIN_DISTINCT_GROUPS,
        model_type=SELECTION_MODEL_TYPE,
        svm_params=SVM_PARAMS,
        mlp_params=active_mlp_params,
        calibration_splits=SVM_CALIBRATION_SPLITS,
        random_state=RANDOM_SEED,
        refit_every=COMBINATION_SELECTION_REFIT_EVERY,
    )
    search_history.to_csv(
        RESULT_DIR / "combination_search.csv",
        index=False,
    )
    if not unconstrained_selected_features:
        raise RuntimeError("최종 Factor가 하나도 선정되지 않았어.")
    (RESULT_DIR / "unconstrained_selected_features.json").write_text(
        json.dumps(unconstrained_selected_features, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # The deployable Logistic champion uses the requested structural core.
    # Transform choice uses only causal predictions ending before the
    # historical research holdout.
    final_core_screen_start = screening_evaluation_start(
        research_y.dropna().index,
        development_end=pre_holdout_end,
        min_train_months=MIN_TRAIN_MONTHS,
        purge_months=FORECAST_HORIZON,
        min_oos_predictions=MIN_OOS_PREDICTIONS,
    )
    final_core_scores = screen_single_factors(
        X=research_X[required_core_columns],
        y=research_y,
        dev_end=pre_holdout_end,
        eval_start=final_core_screen_start,
        min_train=MIN_TRAIN_MONTHS,
        horizon=FORECAST_HORIZON,
        refit_every=SINGLE_FACTOR_REFIT_EVERY,
        n_jobs=-1,
        model_type=SINGLE_FACTOR_MODEL_TYPE,
        svm_params=SVM_PARAMS,
        calibration_splits=SVM_CALIBRATION_SPLITS,
    )
    final_core_scores.to_csv(
        RESULT_DIR / "required_core_final_screen.csv", index=False
    )
    optional_bases = {
        base for bases in optional_families.values() for base in bases
    }
    optional_market_screen = final_core_scores.loc[
        final_core_scores["feature"].str.split("__", n=1).str[0].isin(
            optional_bases
        )
    ].copy()
    optional_market_screen["selection_use"] = (
        "diagnostic_candidate_not_forced_into_required_core"
    )
    optional_market_screen.to_csv(
        RESULT_DIR / "optional_market_factor_screen.csv", index=False
    )
    selected_features, required_core_selection = select_required_core_features(
        final_core_scores,
        core_families,
        min_oos_predictions=MIN_OOS_PREDICTIONS,
        required_transform_tokens=core_transform_tokens,
    )
    required_core_selection.assign(
        selection_cutoff=pre_holdout_end,
        selection_scope="pre_holdout_causal_screen",
    ).to_csv(RESULT_DIR / "required_core_selection.csv", index=False)
    if fixed_primary_features:
        missing_fixed_primary = [
            feature for feature in fixed_primary_features if feature not in research_X
        ]
        unsafe_fixed_primary = [
            feature for feature in fixed_primary_features if feature not in robust_eligible
        ]
        if missing_fixed_primary or unsafe_fixed_primary:
            raise RuntimeError(
                "Predeclared primary features failed availability/safety checks: "
                f"missing={missing_fixed_primary}, unsafe={unsafe_fixed_primary}"
            )
        selected_features = list(fixed_primary_features)
        pd.DataFrame(
            {
                "selection_rank": range(1, len(selected_features) + 1),
                "feature": selected_features,
                "provenance": primary_feature_provenance,
                "target_mode": active_primary_target_spec["mode"],
                "historical_holdout_used": False,
            }
        ).to_csv(RESULT_DIR / "primary_fixed_feature_audit.csv", index=False)

    selected_rows = []
    print()
    print("🏆 최종 Factor")
    for feature in selected_features:
        base = feature.split("__")[0]
        group = feature_groups.get(base, "unknown")
        required_family = next(
            family
            for family, bases in core_families.items()
            if base in bases
        )
        print(f" - [{required_family}/{group}] {feature}")
        selected_rows.append(
            {
                "feature": feature,
                "base": base,
                "group": group,
                "required_family": required_family,
            }
        )
    pd.DataFrame(selected_rows).to_csv(
        RESULT_DIR / "selected_feature_groups.csv",
        index=False,
    )
    feature_group_coverage(factor_candidates, selected_features).to_csv(
        RESULT_DIR / "selected_group_coverage.csv", index=False
    )
    reference_coverage["selected_by_predeclared_research_process"] = (
        reference_coverage["resolved_feature"].isin(selected_features)
    )
    reference_coverage.to_csv(
        RESULT_DIR / "external_reference_coverage.csv", index=False
    )
    original_single_scores = single_scores.loc[
        single_scores["feature"].isin(original_features)
    ].copy()
    original_single_scores.to_csv(
        RESULT_DIR / "original_reference_single_factor_scores.csv", index=False
    )
    original_ranked = original_single_scores.loc[
        (original_single_scores["n"] >= MIN_OOS_PREDICTIONS)
        & original_single_scores["rank_score"].notna()
    ].sort_values("rank_score", ascending=False)["feature"].head(TOP_FEATURE_POOL).tolist()
    original_raw_stage, original_group_stage = build_candidate_funnel(
        original_single_scores.loc[
            original_single_scores["rank_score"].notna()
        ],
        feature_groups,
        max_per_base=RAW_TOP_FEATURES_PER_BASE,
        max_per_group=GROUP_CANDIDATES_PER_GROUP,
    )
    original_raw_stage.to_csv(
        RESULT_DIR / "original_reference_raw_funnel.csv", index=False
    )
    original_ranked = original_group_stage["feature"].tolist()
    original_candidates = prune_correlated_features(
        X=original_research_X,
        ranked_features=original_ranked,
        dev_end=split["dev_end"],
        threshold=CORRELATION_THRESHOLD,
        max_features=EXHAUSTIVE_COMBO_CANDIDATE_POOL,
        min_obs=MIN_OOS_PREDICTIONS,
    )
    original_candidates = round_robin_group_candidates(
        original_candidates,
        feature_groups,
        max_features=EXHAUSTIVE_COMBO_CANDIDATE_POOL,
    )
    original_selected_features, original_search = exhaustive_combination_selection(
        X=original_research_X,
        y=research_y,
        candidates=original_candidates,
        validation_start=split["validation_start"],
        validation_end=split["validation_end"],
        min_train=MIN_TRAIN_MONTHS,
        horizon=FORECAST_HORIZON,
        min_features=MIN_MODEL_FEATURES,
        max_features=EXHAUSTIVE_MAX_COMBO_SIZE,
        max_features_per_base=MAX_FEATURES_PER_BASE,
        feature_groups=feature_groups,
        max_features_per_group=MAX_FEATURES_PER_GROUP,
        min_distinct_groups=MIN_DISTINCT_GROUPS,
        model_type=SELECTION_MODEL_TYPE,
        svm_params=SVM_PARAMS,
        mlp_params=active_mlp_params,
        calibration_splits=SVM_CALIBRATION_SPLITS,
        random_state=RANDOM_SEED,
        refit_every=COMBINATION_SELECTION_REFIT_EVERY,
    )
    if not original_selected_features:
        raise RuntimeError("Original-reference track has no valid exact feature set")
    original_search.to_csv(
        RESULT_DIR / "original_reference_combination_search.csv", index=False
    )
    (RESULT_DIR / "original_reference_selected_features.json").write_text(
        json.dumps(original_selected_features, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with open(
        RESULT_DIR / "selected_features.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            selected_features,
            fp,
            ensure_ascii=False,
            indent=2,
        )

    # Challenger models do not inherit Logistic's final combination.  They
    # re-rank the bounded pre-holdout candidates with their own estimator and
    # perform their own combination search.  The historical holdout remains
    # diagnostic and never feeds this selection.
    model_feature_sets = {"logistic": selected_features}
    challenger_screen_features = group_stage["feature"].tolist()
    for challenger in ("svm", "mlp"):
        if challenger == "mlp" and fixed_mlp_features:
            missing_fixed = [
                feature for feature in fixed_mlp_features if feature not in research_X
            ]
            unsafe_fixed = [
                feature
                for feature in fixed_mlp_features
                if feature not in mlp_deployment_eligible
            ]
            if missing_fixed or unsafe_fixed:
                raise RuntimeError(
                    "Predeclared MLP features failed availability/safety checks: "
                    f"missing={missing_fixed}, unsafe={unsafe_fixed}"
                )
            model_feature_sets["mlp"] = list(fixed_mlp_features)
            fixed_audit = pd.DataFrame(
                {
                    "selection_rank": range(1, len(fixed_mlp_features) + 1),
                    "feature": fixed_mlp_features,
                    "base": [
                        feature.split("__", 1)[0]
                        for feature in fixed_mlp_features
                    ],
                    "selection_method": "predeclared_structural_feature_set",
                    "historical_holdout_used": False,
                }
            )
            fixed_audit.to_csv(
                RESULT_DIR / "mlp_specific_combination_candidates.csv", index=False
            )
            fixed_audit.to_csv(
                RESULT_DIR / "mlp_specific_combination_search.csv", index=False
            )
            fixed_audit.to_csv(
                RESULT_DIR / "mlp_specific_single_factor_scores.csv", index=False
            )
            (RESULT_DIR / "mlp_selected_features.json").write_text(
                json.dumps(list(fixed_mlp_features), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            continue
        model_screen_features = challenger_screen_features
        if challenger == "mlp":
            model_screen_features = [
                feature
                for feature in challenger_screen_features
                if feature in mlp_deployment_eligible
            ]
            if len(model_screen_features) < MIN_MODEL_FEATURES:
                raise RuntimeError(
                    "MLP deployment-safe candidate screen is too small"
                )
        challenger_screening_type = (
            SINGLE_FACTOR_MODEL_TYPE
            if challenger == "mlp"
            else _selection_model_type(challenger)
        )
        challenger_selection_type = _selection_model_type(challenger)
        challenger_kwargs = _model_runtime_kwargs(
            challenger, mlp_params=active_mlp_params
        )
        challenger_scores = screen_single_factors(
            X=research_X[model_screen_features],
            y=research_y,
            dev_end=split["dev_end"],
            eval_start=(split["dev_end"].to_period("M") - 59).to_timestamp("M"),
            min_train=_model_min_train_months(challenger),
            horizon=FORECAST_HORIZON,
            refit_every=COMBINATION_SELECTION_REFIT_EVERY,
            n_jobs=-1,
            model_type=challenger_screening_type,
            **challenger_kwargs,
        )
        challenger_scores.to_csv(
            RESULT_DIR / f"{challenger}_specific_single_factor_scores.csv",
            index=False,
        )
        challenger_usable = challenger_scores.loc[
            (challenger_scores["n"] >= MIN_OOS_PREDICTIONS)
            & challenger_scores["rank_score"].notna()
        ].sort_values("rank_score", ascending=False)
        challenger_ranked = challenger_usable["feature"].tolist()
        challenger_pruned = prune_correlated_features(
            X=research_X,
            ranked_features=challenger_ranked,
            dev_end=split["dev_end"],
            threshold=CORRELATION_THRESHOLD,
            max_features=20,
            min_obs=MIN_OOS_PREDICTIONS,
        )
        challenger_pool = round_robin_group_candidates(
            challenger_pruned,
            feature_groups,
            max_features=min(6, EXHAUSTIVE_COMBO_CANDIDATE_POOL),
        )
        pd.Series(challenger_pool, name="feature").to_csv(
            RESULT_DIR / f"{challenger}_specific_combination_candidates.csv",
            index=False,
        )
        challenger_selected, challenger_history = exhaustive_combination_selection(
            X=research_X,
            y=research_y,
            candidates=challenger_pool,
            validation_start=split["validation_start"],
            validation_end=split["validation_end"],
            min_train=_model_min_train_months(challenger),
            horizon=FORECAST_HORIZON,
            min_features=MIN_MODEL_FEATURES,
            max_features=min(6, EXHAUSTIVE_MAX_COMBO_SIZE),
            max_features_per_base=MAX_FEATURES_PER_BASE,
            feature_groups=feature_groups,
            max_features_per_group=MAX_FEATURES_PER_GROUP,
            min_distinct_groups=MIN_DISTINCT_GROUPS,
            model_type=challenger_selection_type,
            **challenger_kwargs,
            refit_every=COMBINATION_SELECTION_REFIT_EVERY,
        )
        challenger_history.to_csv(
            RESULT_DIR / f"{challenger}_specific_combination_search.csv",
            index=False,
        )
        if not challenger_selected:
            raise RuntimeError(f"{challenger.upper()} challenger has no valid feature set")
        model_feature_sets[challenger] = challenger_selected
        (RESULT_DIR / f"{challenger}_selected_features.json").write_text(
            json.dumps(challenger_selected, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ⑤ HISTORICAL RESEARCH HOLDOUT: CHAMPION AND CHALLENGERS
    print()
    print("⑤ Logistic Champion vs SVM / MLP Challengers")
    economic_review = pd.DataFrame(selected_rows)
    review_registry = pd.read_csv(ECONOMIC_REVIEW_FILE)
    review_columns = [
        "feature",
        "economic_channel",
        "expected_direction",
        "publication_lag_reviewed",
        "duplicate_information_reviewed",
        "review_status",
        "reviewer",
        "reviewed_at",
        "notes",
    ]
    missing_review_columns = set(review_columns).difference(review_registry.columns)
    if missing_review_columns:
        raise ValueError(
            f"economic review registry columns missing: {sorted(missing_review_columns)}"
        )
    economic_review = economic_review.merge(
        review_registry[review_columns],
        on="feature",
        how="left",
        validate="one_to_one",
    )
    economic_review["economic_channel"] = economic_review[
        "economic_channel"
    ].fillna("pending human review")
    economic_review["expected_direction"] = economic_review[
        "expected_direction"
    ].fillna("pending human review")
    for column in ["publication_lag_reviewed", "duplicate_information_reviewed"]:
        economic_review[column] = economic_review[column].map(
            lambda value: str(value).strip().lower() in {"true", "1", "yes", "y"}
            if pd.notna(value)
            else False
        )
    economic_review["review_status"] = economic_review["review_status"].fillna("pending")
    economic_review["review_scope"] = "pre-holdout metadata only"
    economic_review = add_economic_review_drafts(economic_review)
    economic_review.to_csv(RESULT_DIR / "economic_review_log.csv", index=False)

    primary_validation_min_train = (
        MLP_MIN_TRAIN_MONTHS
        if fixed_primary_features
        else _model_min_train_months(FINAL_MODEL_TYPE)
    )
    candidate_folds = make_purged_outer_folds(
        target_index,
        research_end=pre_holdout_end,
        min_train_months=primary_validation_min_train,
        inner_validation_months=INNER_VALIDATION_MONTHS,
        outer_validation_months=OUTER_VALIDATION_MONTHS,
        purge_months=FORECAST_HORIZON,
        screening_oos_months=MIN_OOS_PREDICTIONS,
    )
    if fixed_primary_features:
        fold_availability = fixed_fold_availability_audit(
            X=research_X[list(fixed_primary_features)],
            y=research_y,
            folds=candidate_folds,
            min_train_months=primary_validation_min_train,
            horizon=FORECAST_HORIZON,
        )
    else:
        fold_availability = fold_candidate_availability_audit(
            X=research_X[required_core_columns],
            y=research_y,
            folds=candidate_folds,
            min_train_months=MIN_TRAIN_MONTHS,
            horizon=FORECAST_HORIZON,
            min_oos_predictions=MIN_OOS_PREDICTIONS,
            required_families=core_families,
            required_transform_tokens=core_transform_tokens,
        )
    fold_availability.to_csv(
        RESULT_DIR / "outer_fold_model_availability.csv", index=False
    )
    eligible_fold_ids = set(
        fold_availability.loc[
            fold_availability["fold_availability_passed"], "fold"
        ]
    )
    folds = [fold for fold in candidate_folds if fold.fold in eligible_fold_ids]
    if not folds:
        raise RuntimeError("No Logistic outer fold is model-availability eligible")
    pd.DataFrame([fold.__dict__ for fold in folds]).to_csv(
        RESULT_DIR / "outer_validation_folds.csv", index=False
    )

    if quick:
        print("\nSMOKE TEST: nested feature re-selection skipped")
        quick_eligibility_start = earliest_walk_forward_prediction_date(
            X[selected_features],
            research_y,
            min_train=_model_min_train_months(FINAL_MODEL_TYPE),
            purge=FORECAST_HORIZON,
        )
        nested_prediction = walk_forward_predict(
            X[selected_features],
            research_y,
            eval_start=folds[0].outer_start,
            eval_end=pre_holdout_end,
            min_train=_model_min_train_months(FINAL_MODEL_TYPE),
            purge=FORECAST_HORIZON,
            refit_every=1,
            model_type=FINAL_MODEL_TYPE,
            **_model_runtime_kwargs(
                FINAL_MODEL_TYPE, mlp_params=active_mlp_params
            ),
        )
        nested_selections = pd.DataFrame(
            [
                {
                    "fold": fold.fold,
                    "outer_start": fold.outer_start,
                    "outer_end": fold.outer_end,
                    "selection_rank": rank,
                    "feature": feature,
                    "base": feature.split("__", 1)[0],
                    "group": feature_groups.get(feature.split("__", 1)[0], "unknown"),
                    "model_eligibility_start": quick_eligibility_start,
                    "selection_note": "quick smoke: global pre-holdout selection",
                }
                for fold in folds
                for rank, feature in enumerate(selected_features, start=1)
            ]
        )
        nested_screening = pd.DataFrame()
    elif fixed_primary_features:
        nested_prediction, nested_selections, nested_screening = fixed_outer_predict(
            X=research_X,
            y=research_y,
            folds=folds,
            features=fixed_primary_features,
            feature_groups=feature_groups,
            final_model_type=FINAL_MODEL_TYPE,
            final_min_train_months=primary_validation_min_train,
            horizon=FORECAST_HORIZON,
            refit_every=1,
            svm_params=SVM_PARAMS,
            mlp_params=active_mlp_params,
            calibration_splits=SVM_CALIBRATION_SPLITS,
            random_state=RANDOM_SEED,
            selection_note=(
                f"{primary_feature_provenance};holdout_not_opened"
            ),
        )
    else:
        nested_prediction, nested_selections, nested_screening = nested_outer_predict(
            X=research_X[required_core_columns],
            y=research_y,
            folds=folds,
            feature_groups=feature_groups,
            screening_model_type=SINGLE_FACTOR_MODEL_TYPE,
            selection_model_type=SELECTION_MODEL_TYPE,
            final_model_type=FINAL_MODEL_TYPE,
            min_train_months=MIN_TRAIN_MONTHS,
            final_min_train_months=_model_min_train_months(FINAL_MODEL_TYPE),
            horizon=FORECAST_HORIZON,
            single_factor_refit_every=SINGLE_FACTOR_REFIT_EVERY,
            min_oos_predictions=MIN_OOS_PREDICTIONS,
            top_feature_pool=TOP_FEATURE_POOL,
            raw_top_features_per_base=RAW_TOP_FEATURES_PER_BASE,
            group_candidates_per_group=GROUP_CANDIDATES_PER_GROUP,
            correlation_threshold=CORRELATION_THRESHOLD,
            combination_candidate_pool=NESTED_COMBO_CANDIDATE_POOL,
            exhaustive_candidate_pool=EXHAUSTIVE_COMBO_CANDIDATE_POOL,
            min_model_features=MIN_MODEL_FEATURES,
            max_model_features=MAX_MODEL_FEATURES,
            min_validation_improvement=MIN_VALIDATION_IMPROVEMENT,
            max_features_per_base=MAX_FEATURES_PER_BASE,
            max_features_per_group=MAX_FEATURES_PER_GROUP,
            min_distinct_groups=MIN_DISTINCT_GROUPS,
            svm_params=SVM_PARAMS,
            mlp_params=active_mlp_params,
            calibration_splits=SVM_CALIBRATION_SPLITS,
            random_state=RANDOM_SEED,
            combination_refit_every=COMBINATION_SELECTION_REFIT_EVERY,
            n_jobs=-1,
            required_core_families=core_families,
            required_core_transform_tokens=core_transform_tokens,
        )
    nested_prediction.to_csv(RESULT_DIR / "pre2020_nested_oos_probability.csv")
    nested_selections.to_csv(RESULT_DIR / "outer_fold_feature_selections.csv", index=False)
    nested_screening.to_csv(RESULT_DIR / "outer_fold_screening_top.csv", index=False)
    coefficient_audit = logistic_fold_coefficient_audit(
        X=X,
        y=research_y,
        folds=folds,
        selections=nested_selections,
        horizon=FORECAST_HORIZON,
        min_train_months=_model_min_train_months(FINAL_MODEL_TYPE),
        selection_scope=(
            "quick_global_selection_diagnostic"
            if quick
            else "nested_outer_fold"
        ),
    )
    coefficient_audit.to_csv(
        RESULT_DIR / "outer_fold_logistic_coefficients.csv", index=False
    )
    coefficient_stability = coefficient_sign_stability(coefficient_audit)
    coefficient_stability.to_csv(
        RESULT_DIR / "factor_coefficient_sign_stability.csv", index=False
    )
    family_coefficient_stability = coefficient_family_sign_stability(
        coefficient_audit, core_families
    )
    family_coefficient_stability.to_csv(
        RESULT_DIR / "required_family_coefficient_sign_stability.csv", index=False
    )
    assessable_coefficient_stability = family_coefficient_stability.loc[
        family_coefficient_stability["enough_recurrence_to_assess"]
    ] if not family_coefficient_stability.empty else family_coefficient_stability
    coefficient_sign_consistency_min = (
        float(assessable_coefficient_stability["sign_consistency_ratio"].min())
        if not assessable_coefficient_stability.empty
        else np.nan
    )
    coefficient_sign_consistency_median = (
        float(assessable_coefficient_stability["sign_consistency_ratio"].median())
        if not assessable_coefficient_stability.empty
        else np.nan
    )

    nested_metrics = _model_comparison_row(
        FINAL_MODEL_TYPE, nested_prediction, research_y
    )
    pd.DataFrame([nested_metrics]).to_csv(
        RESULT_DIR / "pre2020_nested_signal_metrics.csv", index=False
    )
    pre_ic_summary, pre_rolling_ic, _ = compute_return_ic(
        signal=nested_prediction,
        future_return=research_future_return,
        rolling_window=IC_ROLLING_WINDOW,
    )
    pd.DataFrame([pre_ic_summary]).to_csv(
        RESULT_DIR / "pre2020_nested_ic_summary.csv", index=False
    )
    pre_rolling_ic.to_csv(RESULT_DIR / "pre2020_nested_rolling_ic.csv")
    fold_signal_rows = _fold_signal_gate(
        folds,
        nested_prediction,
        research_y,
        research_future_return,
        eligibility_starts=_fold_eligibility_map(nested_selections),
    )
    signal_gate_summary, fold_signal = evaluate_signal_gate(
        fold_signal_rows,
        aggregate_auc=nested_metrics["auc"],
        aggregate_rank_ic=pre_ic_summary["rank_ic"],
    )
    fold_signal.to_csv(RESULT_DIR / "outer_fold_signal_metrics.csv", index=False)
    pd.DataFrame([signal_gate_summary]).to_csv(
        RESULT_DIR / "signal_gate_diagnostics.csv", index=False
    )
    fold_signal_pass_ratio = signal_gate_summary[
        "fold_joint_direction_pass_ratio"
    ]
    signal_gate = bool(signal_gate_summary["signal_gate_passed"] and not quick)
    min_train_sensitivity_rows = []
    sensitivity_start = research_X[selected_features].dropna().index.min()
    for min_train_setting in (24, _model_min_train_months(FINAL_MODEL_TYPE)):
        sensitivity_prediction = walk_forward_predict(
            research_X[selected_features],
            research_y,
            eval_start=sensitivity_start,
            eval_end=pre_holdout_end,
            min_train=min_train_setting,
            purge=FORECAST_HORIZON,
            refit_every=1,
            model_type=FINAL_MODEL_TYPE,
            **_model_runtime_kwargs(
                FINAL_MODEL_TYPE, mlp_params=active_mlp_params
            ),
        )
        sensitivity_metrics = evaluate_probabilities(
            sensitivity_prediction, research_y
        )
        sensitivity_ic, _, _ = compute_return_ic(
            sensitivity_prediction,
            target["future_return"].loc[:pre_holdout_end],
            rolling_window=IC_ROLLING_WINDOW,
        )
        min_train_sensitivity_rows.append(
            {
                "min_train_months": min_train_setting,
                **sensitivity_metrics,
                "rank_ic": sensitivity_ic["rank_ic"],
                "pearson_ic": sensitivity_ic["pearson_ic"],
                "selection_use": "original_24m_reproduction_sensitivity_only",
            }
        )
    pd.DataFrame(min_train_sensitivity_rows).to_csv(
        RESULT_DIR / "minimum_training_window_sensitivity.csv", index=False
    )
    effective_sample, nonoverlap = overlapping_target_diagnostics(
        nested_prediction,
        research_y,
        research_future_return,
        horizon=FORECAST_HORIZON,
    )
    effective_sample.to_csv(RESULT_DIR / "effective_sample_size.csv", index=False)
    nonoverlap.to_csv(RESULT_DIR / "nonoverlapping_3m_signal_diagnostics.csv", index=False)
    pre_calibration, pre_reliability = calibration_diagnostics(
        nested_prediction, research_y
    )
    pd.DataFrame([pre_calibration]).to_csv(
        RESULT_DIR / "pre2020_calibration_diagnostics.csv", index=False
    )
    pre_reliability.to_csv(RESULT_DIR / "pre2020_reliability.csv", index=False)
    plot_reliability(
        pre_reliability,
        RESULT_DIR / "pre2020_reliability.png",
        title="Pre-2020 nested OOS reliability",
    )

    sizing_config = {
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
    cash_yield = panel.get("cash_yield_3m")
    fold_labels = pd.Series(np.nan, index=nested_prediction.index)
    for fold in folds:
        fold_labels.loc[fold.outer_start : fold.outer_end] = fold.fold
    pre_comparison, pre_monthly, pre_fold_results = compare_position_sizing(
        market_price=portfolio_market,
        raw_ews=(nested_prediction * 100).rename("raw_ews"),
        cash_yield=cash_yield,
        policies=POSITION_SIZING_POLICIES,
        transaction_cost_scenarios=TRANSACTION_COST_SCENARIOS_BPS,
        sizing_config=sizing_config,
        fold_labels=fold_labels,
        evaluation_end=pre_holdout_end,
    )
    pre_comparison.to_csv(RESULT_DIR / "position_sizing_comparison.csv", index=False)
    pre_monthly.to_csv(RESULT_DIR / "position_sizing_monthly.csv")
    rolling_active_diagnostics(pre_monthly).to_csv(
        RESULT_DIR / "position_sizing_rolling_active.csv", index=False
    )
    pre_fold_results.to_csv(RESULT_DIR / "position_sizing_fold_results.csv", index=False)
    bootstrap = block_bootstrap_policy(
        pre_monthly,
        samples=BOOTSTRAP_SAMPLES,
        block_months=BOOTSTRAP_BLOCK_MONTHS,
        random_seed=RANDOM_SEED,
    )
    bootstrap.to_csv(RESULT_DIR / "position_sizing_bootstrap.csv", index=False)
    selected_policy, policy_decisions = select_position_policy(
        pre_comparison, pre_fold_results, baseline="static_50_50"
    )
    selected_policy, policy_decisions = enforce_signal_gate_fallback(
        selected_policy,
        policy_decisions,
        signal_gate_passed=signal_gate,
    )
    policy_decisions.to_csv(RESULT_DIR / "position_sizing_policy_gate.csv", index=False)
    print(f"\nPre-2020 selected allocation policy: {selected_policy}")

    # MLP is validated independently from the Logistic champion.  The entire
    # screen/select/fit cycle is repeated inside model-specific outer folds;
    # historical holdout observations never participate in this gate.
    print("\n⑤-A MLP nested pre-2020 validation")
    mlp_candidate_folds = make_purged_outer_folds(
        target_index,
        research_end=pre_holdout_end,
        min_train_months=_model_min_train_months("mlp"),
        inner_validation_months=INNER_VALIDATION_MONTHS,
        outer_validation_months=OUTER_VALIDATION_MONTHS,
        purge_months=FORECAST_HORIZON,
        screening_oos_months=MIN_OOS_PREDICTIONS,
    )
    if fixed_mlp_features:
        mlp_fold_availability = fixed_fold_availability_audit(
            X=research_X[list(fixed_mlp_features)],
            y=mlp_research_y,
            folds=mlp_candidate_folds,
            min_train_months=_model_min_train_months("mlp"),
            horizon=FORECAST_HORIZON,
        )
    else:
        mlp_fold_availability = fold_candidate_availability_audit(
            X=mlp_nested_research_X,
            y=mlp_research_y,
            folds=mlp_candidate_folds,
            min_train_months=_model_min_train_months("mlp"),
            horizon=FORECAST_HORIZON,
            min_oos_predictions=MIN_OOS_PREDICTIONS,
            minimum_available_features=MIN_MODEL_FEATURES,
        )
    mlp_fold_availability.to_csv(
        RESULT_DIR / "mlp_outer_fold_model_availability.csv", index=False
    )
    mlp_eligible_fold_ids = set(
        mlp_fold_availability.loc[
            mlp_fold_availability["fold_availability_passed"], "fold"
        ]
    )
    mlp_folds = [
        fold for fold in mlp_candidate_folds if fold.fold in mlp_eligible_fold_ids
    ]
    if not mlp_folds:
        raise RuntimeError("No MLP outer fold is model-availability eligible")
    pd.DataFrame([fold.__dict__ for fold in mlp_folds]).to_csv(
        RESULT_DIR / "mlp_outer_validation_folds.csv", index=False
    )
    if quick:
        mlp_quick_eligibility_start = earliest_walk_forward_prediction_date(
            X[model_feature_sets["mlp"]],
            mlp_research_y,
            min_train=_model_min_train_months("mlp"),
            purge=FORECAST_HORIZON,
        )
        mlp_nested_prediction = walk_forward_predict(
            X[model_feature_sets["mlp"]],
            mlp_research_y,
            eval_start=mlp_folds[0].outer_start,
            eval_end=pre_holdout_end,
            min_train=_model_min_train_months("mlp"),
            purge=FORECAST_HORIZON,
            refit_every=MLP_REFIT_EVERY,
            model_type="mlp",
            **_model_runtime_kwargs("mlp", mlp_params=active_mlp_params),
        )
        mlp_nested_selections = pd.DataFrame(
            [
                {
                    "fold": fold.fold,
                    "outer_start": fold.outer_start,
                    "outer_end": fold.outer_end,
                    "selection_rank": rank,
                    "feature": feature,
                    "base": feature.split("__", 1)[0],
                    "group": feature_groups.get(feature.split("__", 1)[0], "unknown"),
                    "model_eligibility_start": mlp_quick_eligibility_start,
                    "selection_note": "quick smoke: global pre-holdout selection",
                }
                for fold in mlp_folds
                for rank, feature in enumerate(model_feature_sets["mlp"], start=1)
            ]
        )
        mlp_nested_screening = pd.DataFrame()
    elif fixed_mlp_features:
        (
            mlp_nested_prediction,
            mlp_nested_selections,
            mlp_nested_screening,
        ) = fixed_outer_predict(
            X=research_X,
            y=mlp_research_y,
            folds=mlp_folds,
            features=fixed_mlp_features,
            feature_groups=feature_groups,
            final_model_type="mlp",
            final_min_train_months=_model_min_train_months("mlp"),
            horizon=FORECAST_HORIZON,
            refit_every=MLP_REFIT_EVERY,
            mlp_params=active_mlp_params,
            random_state=RANDOM_SEED,
            selection_note=(
                "market_profile_predeclared_structural_set;"
                "outer_outcomes_cannot_change_membership"
            ),
        )
    else:
        mlp_nested_prediction, mlp_nested_selections, mlp_nested_screening = (
            nested_outer_predict(
                X=mlp_nested_research_X,
                y=mlp_research_y,
                folds=mlp_folds,
                feature_groups=feature_groups,
                # A response-local linear funnel keeps the 1,291-column screen
                # tractable; combination selection and every outer prediction
                # still use the MLP itself.
                screening_model_type=SINGLE_FACTOR_MODEL_TYPE,
                selection_model_type="mlp",
                final_model_type="mlp",
                min_train_months=_model_min_train_months("mlp"),
                final_min_train_months=_model_min_train_months("mlp"),
                horizon=FORECAST_HORIZON,
                single_factor_refit_every=SINGLE_FACTOR_REFIT_EVERY,
                min_oos_predictions=MIN_OOS_PREDICTIONS,
                top_feature_pool=TOP_FEATURE_POOL,
                raw_top_features_per_base=RAW_TOP_FEATURES_PER_BASE,
                group_candidates_per_group=GROUP_CANDIDATES_PER_GROUP,
                correlation_threshold=CORRELATION_THRESHOLD,
                combination_candidate_pool=NESTED_COMBO_CANDIDATE_POOL,
                exhaustive_candidate_pool=min(6, EXHAUSTIVE_COMBO_CANDIDATE_POOL),
                min_model_features=MIN_MODEL_FEATURES,
                max_model_features=min(6, MAX_MODEL_FEATURES),
                min_validation_improvement=MIN_VALIDATION_IMPROVEMENT,
                max_features_per_base=MAX_FEATURES_PER_BASE,
                max_features_per_group=MAX_FEATURES_PER_GROUP,
                min_distinct_groups=MIN_DISTINCT_GROUPS,
                svm_params=SVM_PARAMS,
                mlp_params=active_mlp_params,
                calibration_splits=SVM_CALIBRATION_SPLITS,
                random_state=RANDOM_SEED,
                combination_refit_every=COMBINATION_SELECTION_REFIT_EVERY,
                n_jobs=-1,
                final_refit_every=MLP_REFIT_EVERY,
                allow_unavailable_folds=True,
            )
        )
    completed_mlp_fold_ids = set(mlp_nested_selections["fold"].unique())
    unavailable_status = {}
    if (
        not mlp_nested_screening.empty
        and "selection_status" in mlp_nested_screening
    ):
        unavailable_status = (
            mlp_nested_screening.dropna(subset=["selection_status"])
            .groupby("fold")["selection_status"]
            .last()
            .to_dict()
        )
    mlp_fold_selection_status = pd.DataFrame(
        [
            {
                **fold.__dict__,
                "selection_completed": fold.fold in completed_mlp_fold_ids,
                "gate_included": fold.fold in completed_mlp_fold_ids,
                "selection_status": (
                    "selected_and_outer_scored"
                    if fold.fold in completed_mlp_fold_ids
                    else unavailable_status.get(
                        fold.fold, "pre_model_inception_not_evaluable"
                    )
                ),
            }
            for fold in mlp_folds
        ]
    )
    mlp_fold_selection_status.to_csv(
        RESULT_DIR / "mlp_outer_fold_selection_status.csv", index=False
    )
    mlp_folds = [
        fold for fold in mlp_folds if fold.fold in completed_mlp_fold_ids
    ]
    if not mlp_folds:
        raise RuntimeError("No MLP outer fold completed feature selection")
    mlp_nested_prediction.to_csv(
        RESULT_DIR / "mlp_pre2020_nested_oos_probability.csv"
    )
    mlp_nested_selections.to_csv(
        RESULT_DIR / "mlp_outer_fold_feature_selections.csv", index=False
    )
    mlp_nested_screening.to_csv(
        RESULT_DIR / "mlp_outer_fold_screening_top.csv", index=False
    )
    mlp_nested_metrics = _model_comparison_row(
        "mlp", mlp_nested_prediction, mlp_research_y
    )
    pd.DataFrame([mlp_nested_metrics]).to_csv(
        RESULT_DIR / "mlp_pre2020_nested_signal_metrics.csv", index=False
    )
    mlp_pre_ic_summary, mlp_pre_rolling_ic, _ = compute_return_ic(
        signal=mlp_nested_prediction,
        future_return=research_future_return,
        rolling_window=IC_ROLLING_WINDOW,
    )
    pd.DataFrame([mlp_pre_ic_summary]).to_csv(
        RESULT_DIR / "mlp_pre2020_nested_ic_summary.csv", index=False
    )
    mlp_pre_rolling_ic.to_csv(RESULT_DIR / "mlp_pre2020_nested_rolling_ic.csv")
    mlp_fold_signal_rows = _fold_signal_gate(
        mlp_folds,
        mlp_nested_prediction,
        mlp_research_y,
        research_future_return,
        eligibility_starts=_fold_eligibility_map(mlp_nested_selections),
    )
    mlp_signal_gate_summary, mlp_fold_signal = evaluate_signal_gate(
        mlp_fold_signal_rows,
        aggregate_auc=mlp_nested_metrics["auc"],
        aggregate_rank_ic=mlp_pre_ic_summary["rank_ic"],
    )
    mlp_fold_signal.to_csv(
        RESULT_DIR / "mlp_outer_fold_signal_metrics.csv", index=False
    )
    pd.DataFrame([mlp_signal_gate_summary]).to_csv(
        RESULT_DIR / "mlp_signal_gate_diagnostics.csv", index=False
    )
    mlp_signal_gate = bool(
        mlp_signal_gate_summary["signal_gate_passed"] and not quick
    )
    mlp_fold_labels = pd.Series(np.nan, index=mlp_nested_prediction.index)
    for fold in mlp_folds:
        mlp_fold_labels.loc[fold.outer_start : fold.outer_end] = fold.fold
    mlp_pre_comparison, mlp_pre_monthly, mlp_pre_fold_results = compare_position_sizing(
        market_price=portfolio_market,
        raw_ews=(mlp_nested_prediction * 100).rename("raw_ews"),
        cash_yield=cash_yield,
        policies=POSITION_SIZING_POLICIES,
        transaction_cost_scenarios=TRANSACTION_COST_SCENARIOS_BPS,
        sizing_config=sizing_config,
        fold_labels=mlp_fold_labels,
        evaluation_end=pre_holdout_end,
    )
    mlp_pre_comparison.to_csv(
        RESULT_DIR / "mlp_position_sizing_comparison.csv", index=False
    )
    mlp_pre_monthly.to_csv(RESULT_DIR / "mlp_position_sizing_monthly.csv")
    mlp_pre_fold_results.to_csv(
        RESULT_DIR / "mlp_position_sizing_fold_results.csv", index=False
    )
    mlp_selected_policy, mlp_policy_decisions = select_position_policy(
        mlp_pre_comparison,
        mlp_pre_fold_results,
        baseline="static_50_50",
    )
    mlp_selected_policy, mlp_policy_decisions = enforce_signal_gate_fallback(
        mlp_selected_policy,
        mlp_policy_decisions,
        signal_gate_passed=mlp_signal_gate,
    )
    mlp_policy_decisions.to_csv(
        RESULT_DIR / "mlp_position_sizing_policy_gate.csv", index=False
    )
    mlp_selected_policy_decision = mlp_policy_decisions.loc[
        mlp_policy_decisions["policy"].eq(mlp_selected_policy)
    ].iloc[0]
    mlp_allocation_fallback_used = bool(
        mlp_selected_policy_decision.get("selected_as_fallback", False)
    )
    mlp_portfolio_gate = bool(
        mlp_selected_policy_decision["portfolio_gate_passed"] and not quick
    )
    cash_sensitivity_rows = []
    for cash_convention in ("simple_divide_12", "effective_annual_compound"):
        cash_comparison, _, _ = compare_position_sizing(
            market_price=portfolio_market,
            raw_ews=(nested_prediction * 100).rename("raw_ews"),
            cash_yield=cash_yield,
            policies=[selected_policy],
            transaction_cost_scenarios=[TRANSACTION_COST_BPS],
            sizing_config=sizing_config,
            fold_labels=fold_labels,
            evaluation_end=pre_holdout_end,
            cash_return_convention=cash_convention,
        )
        cash_comparison["cash_return_convention"] = cash_convention
        cash_sensitivity_rows.append(cash_comparison)
    pd.concat(cash_sensitivity_rows, ignore_index=True).to_csv(
        RESULT_DIR / "cash_return_convention_sensitivity.csv", index=False
    )

    predictions = {}
    for model_name in EVALUATED_MODELS:
        model_target_y = mlp_y if model_name == "mlp" else y
        predictions[model_name] = walk_forward_predict(
            X[model_feature_sets[model_name]],
            model_target_y,
            eval_start=split["test_start"],
            eval_end=split["test_end"],
            min_train=_model_min_train_months(model_name),
            purge=FORECAST_HORIZON,
            refit_every=(MLP_REFIT_EVERY if model_name == "mlp" else 1),
            model_type=model_name,
            **_model_runtime_kwargs(
                model_name, mlp_params=active_mlp_params
            ),
        )
    original_reference_prediction = walk_forward_predict(
        X[original_selected_features],
        y,
        eval_start=split["test_start"],
        eval_end=split["test_end"],
        min_train=_model_min_train_months(FINAL_MODEL_TYPE),
        purge=FORECAST_HORIZON,
        refit_every=1,
        model_type=FINAL_MODEL_TYPE,
        **_model_runtime_kwargs(
            FINAL_MODEL_TYPE, mlp_params=active_mlp_params
        ),
    )
    original_reference_prediction.to_csv(
        RESULT_DIR / "original_reference_holdout_probability.csv"
    )
    sizing_seed_predictions = {}
    for model_name in predictions:
        model_target_y = mlp_y if model_name == "mlp" else y
        sizing_seed_predictions[model_name] = walk_forward_predict(
            X[model_feature_sets[model_name]],
            model_target_y,
            eval_start=pre_holdout_end,
            eval_end=pre_holdout_end,
            min_train=_model_min_train_months(model_name),
            purge=FORECAST_HORIZON,
            refit_every=(MLP_REFIT_EVERY if model_name == "mlp" else 1),
            model_type=model_name,
            **_model_runtime_kwargs(
                model_name, mlp_params=active_mlp_params
            ),
        )
    original_seed_prediction = walk_forward_predict(
        X[original_selected_features],
        y,
        eval_start=pre_holdout_end,
        eval_end=pre_holdout_end,
        min_train=_model_min_train_months(FINAL_MODEL_TYPE),
        purge=FORECAST_HORIZON,
        refit_every=1,
        model_type=FINAL_MODEL_TYPE,
        **_model_runtime_kwargs(
            FINAL_MODEL_TYPE, mlp_params=active_mlp_params
        ),
    )

    # 세 모델 비교도 동일한 예측 가능 월에서만 수행한다.
    common_prediction_index = predictions[EVALUATED_MODELS[0]].dropna().index
    for model_name in EVALUATED_MODELS[1:]:
        common_prediction_index = common_prediction_index.intersection(
            predictions[model_name].dropna().index
        )
    if len(common_prediction_index) == 0:
        raise RuntimeError("Logistic/SVM/MLP 공통 Test 예측기간이 없어.")
    predictions = {
        name: prediction.loc[common_prediction_index]
        for name, prediction in predictions.items()
    }

    for name, prediction in predictions.items():
        ews = (prediction * 100).rename("ews")
        ews.to_csv(RESULT_DIR / f"{name}_test_ews.csv")

    portfolio_prediction_end = min(
        _last_completed_month(market.index),
        _last_completed_month(portfolio_market.index),
    )
    portfolio_predictions = {}
    for model_name, prediction in predictions.items():
        model_target_y = mlp_y if model_name == "mlp" else y
        portfolio_predictions[model_name] = extend_prediction_tail(
            prediction,
            X[model_feature_sets[model_name]],
            model_target_y,
            eval_end=portfolio_prediction_end,
            min_train=_model_min_train_months(model_name),
            purge=FORECAST_HORIZON,
            refit_every=(MLP_REFIT_EVERY if model_name == "mlp" else 1),
            model_type=model_name,
            model_kwargs=_model_runtime_kwargs(
                model_name, mlp_params=active_mlp_params
            ),
        )
        (portfolio_predictions[model_name] * 100).rename("ews").to_csv(
            RESULT_DIR / f"{model_name}_portfolio_ews.csv"
        )
    original_reference_portfolio_prediction = extend_prediction_tail(
        original_reference_prediction,
        X[original_selected_features],
        y,
        eval_end=portfolio_prediction_end,
        min_train=_model_min_train_months(FINAL_MODEL_TYPE),
        purge=FORECAST_HORIZON,
        refit_every=1,
        model_type=FINAL_MODEL_TYPE,
        model_kwargs=_model_runtime_kwargs(
            FINAL_MODEL_TYPE, mlp_params=active_mlp_params
        ),
    )

    model_comparison = pd.DataFrame(
        [
            _model_comparison_row(
                name,
                prediction,
                mlp_y if name == "mlp" else y,
            )
            for name, prediction in predictions.items()
        ]
    )
    model_comparison["target_mode"] = model_comparison["model"].map(
        {
            "Logistic": active_primary_target_spec["mode"],
            "SVM": active_primary_target_spec["mode"],
            "MLP": active_mlp_target_spec["mode"],
        }
    )
    model_comparison.to_csv(
        RESULT_DIR / "model_comparison.csv",
        index=False,
    )
    model_comparison[
        [
            "model",
            "target_mode",
            "n",
            "auc",
            "brier",
            "accuracy",
            "risk_on_rate",
            "naive_accuracy",
            "accuracy_lift_vs_naive",
        ]
    ].to_csv(RESULT_DIR / "class_balance.csv", index=False)

    print()
    print("===== MODEL METRICS + CLASS BALANCE =====")
    print(
        model_comparison[
            [
                "model",
                "target_mode",
                "n",
                "auc",
                "brier",
                "accuracy",
                "risk_on_rate",
                "naive_accuracy",
                "accuracy_lift_vs_naive",
            ]
        ].to_string(index=False)
    )

    for row in model_comparison.itertuples(index=False):
        if np.isfinite(row.auc) and row.auc < 0.5:
            print(
                f"[WARN] {row.model}: Test AUC {row.auc:.4f} < 0.5. "
                "방향 이상 진단만 기록하며 예측을 뒤집지 않아."
            )
        if (
            np.isfinite(row.accuracy_lift_vs_naive)
            and row.accuracy_lift_vs_naive <= 0
        ):
            print(
                f"[WARN] {row.model}: 정확도가 다수 class 기준선을 "
                "넘지 못했어."
            )

    if FINAL_MODEL_TYPE not in predictions:
        raise ValueError(
            f"지원하지 않는 FINAL_MODEL_TYPE: {FINAL_MODEL_TYPE}"
        )
    final_label = MODEL_LABELS[FINAL_MODEL_TYPE]
    final_prediction = predictions[FINAL_MODEL_TYPE]
    final_metrics = model_comparison.loc[
        model_comparison["model"] == final_label
    ].iloc[0]
    final_metrics.to_frame().T.to_csv(
        RESULT_DIR / "test_metrics.csv",
        index=False,
    )
    (final_prediction * 100).rename("ews").to_csv(
        RESULT_DIR / "test_ews.csv"
    )

    holdout_calibration, holdout_reliability = calibration_diagnostics(
        final_prediction, y
    )
    pd.DataFrame([holdout_calibration]).to_csv(
        RESULT_DIR / "calibration_diagnostics.csv", index=False
    )
    holdout_reliability.to_csv(
        RESULT_DIR / "calibration_reliability.csv", index=False
    )
    plot_reliability(
        holdout_reliability,
        RESULT_DIR / "calibration_reliability.png",
        title="Historical research holdout reliability",
    )

    combined_sizing_scores = pd.concat(
        [
            nested_prediction.loc[: pre_holdout_end - pd.offsets.MonthEnd(1)],
            sizing_seed_predictions[FINAL_MODEL_TYPE],
            portfolio_predictions[FINAL_MODEL_TYPE],
        ]
    ).sort_index()
    combined_sizing_scores = combined_sizing_scores[
        ~combined_sizing_scores.index.duplicated(keep="last")
    ]
    holdout_comparison, holdout_monthly, holdout_fold_results = (
        compare_position_sizing(
            market_price=portfolio_market,
            raw_ews=(combined_sizing_scores * 100).rename("raw_ews"),
            cash_yield=cash_yield,
            policies=POSITION_SIZING_POLICIES,
            transaction_cost_scenarios=TRANSACTION_COST_SCENARIOS_BPS,
            sizing_config=sizing_config,
            evaluation_start=split["test_start"],
            evaluation_end=portfolio_prediction_end,
        )
    )
    holdout_comparison["selection_use"] = "diagnostic_only_not_used_for_selection"
    holdout_comparison.to_csv(
        RESULT_DIR / "research_holdout_report.csv", index=False
    )
    holdout_monthly.to_csv(
        RESULT_DIR / "research_holdout_position_sizing_monthly.csv"
    )
    original_raw_ews = pd.concat(
        [original_seed_prediction, original_reference_portfolio_prediction]
    ).sort_index() * 100
    original_target_weight = target_weight_from_ews(
        original_raw_ews, policy="linear", **sizing_config
    )
    original_backtest = run_backtest(
        market_price=portfolio_market,
        ews=original_raw_ews,
        target_stock_weight=original_target_weight,
        allocation_policy="linear",
        cash_yield_annual_pct=cash_yield,
        min_stock_weight=MIN_STOCK_WEIGHT,
        max_stock_weight=MAX_STOCK_WEIGHT,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        verbose=False,
        market_name=portfolio_instrument,
    )
    original_backtest.to_csv(
        RESULT_DIR / "original_reference_holdout_backtest.csv"
    )
    original_rows = evaluate_backtest(
        original_backtest,
        f"Original-reference {final_label} linear",
        allocation_policy="linear",
        benchmark_name=portfolio_instrument,
    )
    original_signal_metrics = _model_comparison_row(
        FINAL_MODEL_TYPE, original_reference_prediction, y
    )
    pd.DataFrame(
        [{**original_signal_metrics, "selection_use": "diagnostic_only"}]
    ).to_csv(RESULT_DIR / "original_reference_holdout_signal.csv", index=False)
    pd.DataFrame(original_rows).assign(
        selection_use="diagnostic_only"
    ).to_csv(
        RESULT_DIR / "original_reference_holdout_performance.csv", index=False
    )
    rolling_active_diagnostics(holdout_monthly).to_csv(
        RESULT_DIR / "research_holdout_rolling_active.csv", index=False
    )

    # ⑥ MODEL-SPECIFIC BACKTESTS
    print()
    print("⑥ Logistic / SVM / MLP 개별 백테스트")
    backtests = {}
    for name, prediction in portfolio_predictions.items():
        raw_ews = (prediction * 100).rename("raw_ews")
        model_nested_history = (
            mlp_nested_prediction if name == "mlp" else nested_prediction
        )
        model_allocation_policy = (
            mlp_selected_policy if name == "mlp" else selected_policy
        )
        sizing_history = pd.concat(
            [
                model_nested_history.loc[
                    : pre_holdout_end - pd.offsets.MonthEnd(1)
                ] * 100,
                sizing_seed_predictions[name] * 100,
                raw_ews,
            ]
        ).sort_index()
        sizing_history = sizing_history[
            ~sizing_history.index.duplicated(keep="last")
        ]
        target_weight = target_weight_from_ews(
            sizing_history,
            policy=model_allocation_policy,
            **sizing_config,
        ).loc[pre_holdout_end:]
        backtest_raw_ews = pd.concat(
            [sizing_seed_predictions[name] * 100, raw_ews]
        ).sort_index()
        backtest = run_backtest(
            market_price=portfolio_market,
            ews=backtest_raw_ews,
            target_stock_weight=target_weight,
            allocation_policy=model_allocation_policy,
            cash_yield_annual_pct=cash_yield,
            min_stock_weight=MIN_STOCK_WEIGHT,
            max_stock_weight=MAX_STOCK_WEIGHT,
            transaction_cost_bps=TRANSACTION_COST_BPS,
            market_name=portfolio_instrument,
        )
        backtests[name] = backtest
        backtest.to_csv(RESULT_DIR / f"{name}_backtest.csv")

    # 두 모델 및 네 전략 모두 실제 거래 가능한 완전 동일 기간으로 제한.
    common_evaluation_index = backtest_evaluation_window(
        backtests[EVALUATED_MODELS[0]]
    ).index
    for model_name in EVALUATED_MODELS[1:]:
        common_evaluation_index = common_evaluation_index.intersection(
            backtest_evaluation_window(backtests[model_name]).index
        )
    if len(common_evaluation_index) == 0:
        raise RuntimeError("Logistic/SVM/MLP 공통 백테스트 기간이 없어.")

    performance_rows = []
    for name, backtest in backtests.items():
        performance_rows.extend(
            evaluate_backtest(
                backtest,
                MODEL_LABELS[name],
                evaluation_index=common_evaluation_index,
                allocation_policy=(
                    mlp_selected_policy if name == "mlp" else selected_policy
                ),
                benchmark_name=portfolio_instrument,
            )
        )
    performance = pd.DataFrame(performance_rows)
    if performance["Months"].nunique() != 1:
        raise AssertionError(
            "모든 전략의 평가 개월 수가 같지 않아. 비교를 중단해."
        )
    performance.to_csv(
        RESULT_DIR / "performance_comparison.csv",
        index=False,
    )
    plot_model_comparison(
        model_comparison,
        performance,
        RESULT_DIR / "model_comparison.png",
    )

    print()
    print(
        "성과평가 공통기간:",
        common_evaluation_index.min().date(),
        "~",
        common_evaluation_index.max().date(),
    )
    print("공통 평가 개월:", len(common_evaluation_index))
    print("=" * 125)
    print("📈 SAME-PERIOD STRATEGY PERFORMANCE")
    print("=" * 125)
    print(performance.to_string(index=False))

    final_backtest = backtests[FINAL_MODEL_TYPE]
    final_backtest.to_csv(RESULT_DIR / "backtest.csv")
    final_strategy_row = performance.loc[
        (performance["model"] == final_label)
        & (performance["strategy"] == f"{final_label} Dynamic")
    ].iloc[0]
    performance.loc[
        performance["model"] == final_label
    ].to_csv(RESULT_DIR / "backtest_stats.csv", index=False)

    # ⑦ RETURN IC: 모델별로 따로 계산한다.
    print()
    print("⑦ MODEL-SPECIFIC RETURN IC")
    ic_rows = []
    rolling_ics = {}
    for name, prediction in predictions.items():
        summary, rolling_ic, _ = compute_return_ic(
            signal=prediction,
            future_return=target["future_return"],
            rolling_window=IC_ROLLING_WINDOW,
        )
        rolling_ics[name] = rolling_ic
        ic_rows.append({"model": MODEL_LABELS[name], **summary})
        rolling_ic.to_csv(RESULT_DIR / f"{name}_rolling_ic.csv")

    ic_comparison = pd.DataFrame(ic_rows)
    ic_comparison.to_csv(
        RESULT_DIR / "model_ic_comparison.csv",
        index=False,
    )
    print(ic_comparison.to_string(index=False))

    # Preserve the 2020+ MLP evidence as a clearly labelled final diagnostic.
    # It can confirm a feature/policy specification that was locked on the
    # pre-2020 development sample, but it may not change that specification.
    mlp_holdout_performance = performance.loc[
        performance["model"].eq("MLP")
    ].copy()
    mlp_holdout_performance["selection_use"] = (
        "independent_historical_holdout_confirmation_not_selection"
    )
    mlp_holdout_performance.to_csv(
        RESULT_DIR / "mlp_historical_holdout_performance.csv", index=False
    )
    mlp_holdout_signal = model_comparison.loc[
        model_comparison["model"].eq("MLP")
    ].merge(
        ic_comparison.loc[
            ic_comparison["model"].eq("MLP"),
            ["model", "pearson_ic", "rank_ic"],
        ],
        on="model",
        how="left",
        validate="one_to_one",
    )
    mlp_holdout_signal["selection_use"] = (
        "independent_historical_holdout_confirmation_not_selection"
    )
    mlp_holdout_signal.to_csv(
        RESULT_DIR / "mlp_historical_holdout_signal.csv", index=False
    )
    mlp_holdout_dynamic = mlp_holdout_performance.loc[
        mlp_holdout_performance["strategy"].eq("MLP Dynamic")
    ].iloc[0]
    mlp_holdout_same_exposure = mlp_holdout_performance.loc[
        mlp_holdout_performance["strategy"].str.startswith("MLP Same Exposure")
    ].iloc[0]
    # The holdout is not a source of new features, hyperparameters or policy
    # choice.  It is nevertheless allowed to veto capital deployment when the
    # locked tactical overlay is plainly worse than same exposure.  Passing
    # this check cannot by itself promote a model; failing it forces static.
    (
        mlp_holdout_safety_gate,
        mlp_holdout_safety_checks,
        mlp_holdout_safety_differences,
    ) = evaluate_holdout_safety_veto(
        auc=mlp_holdout_signal.iloc[0]["auc"],
        rank_ic=mlp_holdout_signal.iloc[0]["rank_ic"],
        dynamic_sharpe=mlp_holdout_dynamic["Sharpe"],
        same_exposure_sharpe=mlp_holdout_same_exposure["Sharpe"],
        dynamic_cagr=mlp_holdout_dynamic["CAGR"],
        same_exposure_cagr=mlp_holdout_same_exposure["CAGR"],
        dynamic_max_drawdown=mlp_holdout_dynamic["MaxDrawdown"],
        same_exposure_max_drawdown=mlp_holdout_same_exposure["MaxDrawdown"],
    )
    mlp_holdout_sharpe_difference = mlp_holdout_safety_differences[
        "sharpe_difference"
    ]
    mlp_holdout_cagr_difference = mlp_holdout_safety_differences[
        "cagr_difference"
    ]
    mlp_holdout_drawdown_difference = mlp_holdout_safety_differences[
        "drawdown_difference"
    ]
    mlp_pre_veto_selected_policy = mlp_selected_policy
    pd.DataFrame(
        [
            {
                "feature_set_provenance": fixed_mlp_feature_provenance,
                "target_mode": active_mlp_target_spec["mode"],
                "target_spec_json": json.dumps(
                    active_mlp_target_spec, sort_keys=True
                ),
                "feature_membership_locked_before_holdout": bool(
                    fixed_mlp_features
                ),
                "holdout_used_for_feature_or_policy_selection": False,
                "used_for_positive_promotion": False,
                "used_as_fail_closed_safety_veto": True,
                "start": mlp_holdout_dynamic["Start"],
                "end": mlp_holdout_dynamic["End"],
                "months": mlp_holdout_dynamic["Months"],
                "evaluated_locked_allocation_policy": (
                    mlp_pre_veto_selected_policy
                ),
                "transaction_cost_bps": TRANSACTION_COST_BPS,
                "auc": mlp_holdout_signal.iloc[0]["auc"],
                "rank_ic": mlp_holdout_signal.iloc[0]["rank_ic"],
                "dynamic_cagr": mlp_holdout_dynamic["CAGR"],
                "same_exposure_cagr": mlp_holdout_same_exposure["CAGR"],
                "cagr_difference": mlp_holdout_cagr_difference,
                "dynamic_sharpe": mlp_holdout_dynamic["Sharpe"],
                "same_exposure_sharpe": mlp_holdout_same_exposure["Sharpe"],
                "sharpe_difference": mlp_holdout_sharpe_difference,
                "dynamic_max_drawdown": mlp_holdout_dynamic["MaxDrawdown"],
                "same_exposure_max_drawdown": mlp_holdout_same_exposure[
                    "MaxDrawdown"
                ],
                "drawdown_difference": mlp_holdout_drawdown_difference,
                **mlp_holdout_safety_checks,
                "holdout_safety_gate": mlp_holdout_safety_gate,
                "interpretation": (
                    "passing cannot promote; failure vetoes tactical capital use; "
                    "next untouched observations belong in the frozen shadow ledger"
                ),
            }
        ]
    ).to_csv(
        RESULT_DIR / "mlp_historical_holdout_confirmation.csv", index=False
    )
    mlp_policy_decisions["selected_before_holdout_safety_veto"] = (
        mlp_policy_decisions["policy"].eq(mlp_pre_veto_selected_policy)
    )
    if not mlp_holdout_safety_gate:
        mlp_policy_decisions["selected"] = False
        fallback_mask = mlp_policy_decisions["policy"].eq("static_50_50")
        if fallback_mask.sum() != 1:
            raise RuntimeError("MLP static fail-closed policy is ambiguous")
        mlp_policy_decisions.loc[fallback_mask, "selected"] = True
        mlp_policy_decisions.loc[fallback_mask, "selected_as_fallback"] = True
        mlp_policy_decisions.loc[fallback_mask, "selection_reason"] = (
            "historical_holdout_safety_veto"
        )
        mlp_selected_policy = "static_50_50"
        mlp_allocation_fallback_used = True
    mlp_portfolio_gate = bool(
        mlp_portfolio_gate and mlp_holdout_safety_gate
    )
    mlp_policy_decisions.to_csv(
        RESULT_DIR / "mlp_position_sizing_policy_gate.csv", index=False
    )

    final_ic = ic_comparison.loc[
        ic_comparison["model"] == final_label
    ].iloc[0]
    final_ic.to_frame().T.to_csv(
        RESULT_DIR / "ic_summary.csv",
        index=False,
    )
    final_rolling_ic = rolling_ics[FINAL_MODEL_TYPE]
    final_rolling_ic.to_csv(RESULT_DIR / "rolling_ic.csv")

    rolling_sr = rolling_sharpe(
        returns=final_backtest["strategy_return"],
        risk_free=final_backtest["cash_return"],
        window=SHARPE_ROLLING_WINDOW,
    )
    rolling_sr.to_csv(RESULT_DIR / "rolling_sharpe.csv")

    selected_policy_decision = policy_decisions.loc[
        policy_decisions["policy"].eq(selected_policy)
    ].iloc[0]
    portfolio_gate = bool(selected_policy_decision["portfolio_gate_passed"])
    allocation_fallback_used = bool(
        selected_policy_decision.get("selected_as_fallback", False)
    )
    primary_pre_veto_selected_policy = selected_policy
    primary_holdout_signal = model_comparison.loc[
        model_comparison["model"].eq(final_label)
    ].iloc[0]
    primary_holdout_dynamic = performance.loc[
        (performance["model"].eq(final_label))
        & (performance["strategy"].eq(f"{final_label} Dynamic"))
    ].iloc[0]
    primary_holdout_same_exposure = performance.loc[
        (performance["model"].eq(final_label))
        & performance["strategy"].str.startswith(
            f"{final_label} Same Exposure"
        )
    ].iloc[0]
    (
        primary_holdout_safety_gate,
        primary_holdout_safety_checks,
        primary_holdout_safety_differences,
    ) = evaluate_holdout_safety_veto(
        auc=primary_holdout_signal["auc"],
        rank_ic=final_ic["rank_ic"],
        dynamic_sharpe=primary_holdout_dynamic["Sharpe"],
        same_exposure_sharpe=primary_holdout_same_exposure["Sharpe"],
        dynamic_cagr=primary_holdout_dynamic["CAGR"],
        same_exposure_cagr=primary_holdout_same_exposure["CAGR"],
        dynamic_max_drawdown=primary_holdout_dynamic["MaxDrawdown"],
        same_exposure_max_drawdown=primary_holdout_same_exposure["MaxDrawdown"],
    )
    pd.DataFrame(
        [
            {
                "model": final_label,
                "target_mode": active_primary_target_spec["mode"],
                "target_spec_json": json.dumps(
                    active_primary_target_spec, sort_keys=True
                ),
                "holdout_used_for_feature_or_policy_selection": False,
                "used_for_positive_promotion": False,
                "used_as_fail_closed_safety_veto": True,
                "evaluated_locked_allocation_policy": (
                    primary_pre_veto_selected_policy
                ),
                "start": primary_holdout_dynamic["Start"],
                "end": primary_holdout_dynamic["End"],
                "months": primary_holdout_dynamic["Months"],
                "auc": primary_holdout_signal["auc"],
                "rank_ic": final_ic["rank_ic"],
                "dynamic_cagr": primary_holdout_dynamic["CAGR"],
                "same_exposure_cagr": primary_holdout_same_exposure["CAGR"],
                "dynamic_sharpe": primary_holdout_dynamic["Sharpe"],
                "same_exposure_sharpe": primary_holdout_same_exposure["Sharpe"],
                "dynamic_max_drawdown": primary_holdout_dynamic["MaxDrawdown"],
                "same_exposure_max_drawdown": primary_holdout_same_exposure[
                    "MaxDrawdown"
                ],
                **primary_holdout_safety_differences,
                **primary_holdout_safety_checks,
                "holdout_safety_gate": primary_holdout_safety_gate,
            }
        ]
    ).to_csv(RESULT_DIR / "historical_holdout_confirmation.csv", index=False)
    if not primary_holdout_safety_gate:
        policy_decisions["selected"] = False
        fallback_mask = policy_decisions["policy"].eq("static_50_50")
        if fallback_mask.sum() != 1:
            raise RuntimeError("Primary static fail-closed policy is ambiguous")
        policy_decisions.loc[fallback_mask, "selected"] = True
        policy_decisions.loc[fallback_mask, "selected_as_fallback"] = True
        policy_decisions.loc[fallback_mask, "selection_reason"] = (
            "historical_holdout_safety_veto"
        )
        selected_policy = "static_50_50"
        allocation_fallback_used = True
    portfolio_gate = bool(portfolio_gate and primary_holdout_safety_gate)
    policy_decisions.to_csv(
        RESULT_DIR / "position_sizing_policy_gate.csv", index=False
    )

    operational_source_features = [
        *selected_features,
        "cash_yield_3m__level",
    ]
    selected_vintage = selected_point_in_time_audit(
        operational_source_features,
        raw_catalog,
        point_in_time,
        feature_metadata=factor_candidates,
    )
    selected_vintage["usage_role"] = np.where(
        selected_vintage["base"].eq("cash_yield_3m"),
        "portfolio_cash_return",
        "model_feature",
    )
    selected_vintage.to_csv(
        RESULT_DIR / "selected_source_point_in_time_audit.csv", index=False
    )
    point_in_time_vintage_gate = bool(
        selected_vintage["strict_vintage_gate_passed"].all()
    )
    release_timing_gate = bool(
        selected_vintage["release_timing_gate_passed"].all()
    )
    investable_return_source_gate = bool(
        market_roles.loc[
            market_roles["role"].eq("investable_portfolio_return"),
            "deployment_eligible",
        ].all()
    )
    mlp_operational_source_features = [
        *model_feature_sets["mlp"],
        "cash_yield_3m__level",
    ]
    mlp_selected_vintage = selected_point_in_time_audit(
        mlp_operational_source_features,
        raw_catalog,
        point_in_time,
        feature_metadata=factor_candidates,
    )
    mlp_selected_vintage["usage_role"] = np.where(
        mlp_selected_vintage["base"].eq("cash_yield_3m"),
        "portfolio_cash_return",
        "model_feature",
    )
    mlp_selected_vintage.to_csv(
        RESULT_DIR / "mlp_selected_source_point_in_time_audit.csv", index=False
    )
    mlp_point_in_time_vintage_gate = bool(
        mlp_selected_vintage["strict_vintage_gate_passed"].all()
    )
    mlp_release_timing_gate = bool(
        mlp_selected_vintage["release_timing_gate_passed"].all()
    )
    mlp_review_rows = pd.DataFrame(
        [
            {
                "feature": feature,
                "base": feature.split("__", 1)[0],
                "group": feature_groups.get(feature.split("__", 1)[0], "unknown"),
            }
            for feature in model_feature_sets["mlp"]
        ]
    )
    mlp_economic_review = mlp_review_rows.merge(
        review_registry[review_columns],
        on="feature",
        how="left",
        validate="one_to_one",
    )
    mlp_economic_review["economic_channel"] = mlp_economic_review[
        "economic_channel"
    ].fillna("pending human review")
    mlp_economic_review["expected_direction"] = mlp_economic_review[
        "expected_direction"
    ].fillna("pending human review")
    for column in ["publication_lag_reviewed", "duplicate_information_reviewed"]:
        mlp_economic_review[column] = mlp_economic_review[column].map(
            lambda value: str(value).strip().lower() in {"true", "1", "yes", "y"}
            if pd.notna(value)
            else False
        )
    mlp_economic_review["review_status"] = mlp_economic_review[
        "review_status"
    ].fillna("pending")
    mlp_economic_review = add_economic_review_drafts(mlp_economic_review)
    mlp_economic_review.assign(
        review_scope="pre-holdout metadata only",
        approval_gate=mlp_economic_review["review_status"].eq("approved"),
        strict_deployment_blocker=lambda frame: ~(
            frame["review_status"].eq("approved")
            & frame["publication_lag_reviewed"]
            & frame["duplicate_information_reviewed"]
        ),
    ).to_csv(RESULT_DIR / "mlp_economic_review_checklist.csv", index=False)
    mlp_economic_review_approved_gate = bool(
        mlp_economic_review["review_status"].eq("approved").all()
    )
    mlp_publication_lag_review_gate = bool(
        mlp_economic_review["publication_lag_reviewed"].all()
    )
    mlp_duplicate_information_review_gate = bool(
        mlp_economic_review["duplicate_information_reviewed"].all()
    )
    mlp_strict_operational_gate = bool(
        not quick
        and freshness_audit["schema_passed"].all()
        and freshness_audit["freshness_passed"].all()
        and mlp_point_in_time_vintage_gate
        and mlp_release_timing_gate
        and investable_return_source_gate
        and mlp_economic_review_approved_gate
        and mlp_publication_lag_review_gate
        and mlp_duplicate_information_review_gate
    )
    mlp_capital_use_allowed = bool(
        mlp_signal_gate and mlp_portfolio_gate and mlp_strict_operational_gate
    )
    mlp_validation_row = {
        "model": "MLP",
        "validation_scope": (
            "pre2020_locked_feature_outer_score"
            if fixed_mlp_features
            else "pre2020_nested_outer"
        ),
        "feature_membership_locked_before_outer_scoring": bool(
            fixed_mlp_features
        ),
        "feature_set_provenance": fixed_mlp_feature_provenance,
        "historical_holdout_used_for_selection": False,
        "historical_holdout_used_for_positive_promotion": False,
        "historical_holdout_used_as_safety_veto": True,
        "historical_holdout_confirmation_file": (
            "mlp_historical_holdout_confirmation.csv"
        ),
        "historical_holdout_safety_gate": mlp_holdout_safety_gate,
        "model_refit_every_months": MLP_REFIT_EVERY,
        "target_mode": active_mlp_target_spec["mode"],
        "target_spec_json": json.dumps(active_mlp_target_spec, sort_keys=True),
        "model_params_json": json.dumps(active_mlp_params, sort_keys=True),
        "deployment_safe_candidate_count": len(mlp_compact_features),
        "model_inception": (
            str(pd.to_datetime(
                mlp_nested_selections["model_eligibility_start"]
            ).min().date())
            if not mlp_nested_selections.empty
            and "model_eligibility_start" in mlp_nested_selections
            else None
        ),
        "aggregate_auc": float(mlp_signal_gate_summary["aggregate_auc"]),
        "aggregate_rank_ic": float(mlp_signal_gate_summary["aggregate_rank_ic"]),
        "fold_joint_direction_pass_ratio": float(
            mlp_signal_gate_summary["fold_joint_direction_pass_ratio"]
        ),
        "signal_gate": mlp_signal_gate,
        "allocation_policy": mlp_selected_policy,
        "tactical_overlay_active": not mlp_allocation_fallback_used,
        "fail_closed_fallback_used": mlp_allocation_fallback_used,
        "portfolio_gate": mlp_portfolio_gate,
        "point_in_time_vintage_gate": mlp_point_in_time_vintage_gate,
        "release_timing_gate": mlp_release_timing_gate,
        "investable_return_source_gate": investable_return_source_gate,
        "economic_review_approved_gate": mlp_economic_review_approved_gate,
        "publication_lag_review_gate": mlp_publication_lag_review_gate,
        "duplicate_information_review_gate": mlp_duplicate_information_review_gate,
        "strict_operational_gate": mlp_strict_operational_gate,
        "promotion_gate": mlp_capital_use_allowed,
        "research_shadow_allowed": True,
        "capital_use_allowed": mlp_capital_use_allowed,
        "failed_conditions": "|".join(
            condition
            for condition, passed in (
                ("signal_gate", mlp_signal_gate),
                ("portfolio_gate", mlp_portfolio_gate),
                (
                    "historical_holdout_safety_gate",
                    mlp_holdout_safety_gate,
                ),
                ("strict_operational_gate", mlp_strict_operational_gate),
            )
            if not passed
        ),
    }
    pd.DataFrame([mlp_validation_row]).to_csv(
        RESULT_DIR / "mlp_validation_gates.csv", index=False
    )
    economic_review_approved_gate = bool(
        economic_review["review_status"].eq("approved").all()
    )
    publication_lag_review_gate = bool(
        economic_review["publication_lag_reviewed"].all()
    )
    duplicate_information_review_gate = bool(
        economic_review["duplicate_information_reviewed"].all()
    )
    economic_review.assign(
        approval_gate=economic_review["review_status"].eq("approved"),
        strict_deployment_blocker=lambda frame: ~(
            frame["review_status"].eq("approved")
            & frame["publication_lag_reviewed"]
            & frame["duplicate_information_reviewed"]
        ),
        next_action=(
            "human reviewer must document economic channel, expected direction, "
            "release lag and duplicate-information review"
        ),
    ).to_csv(RESULT_DIR / "economic_review_checklist.csv", index=False)

    operational_core_gate = bool(
        not quick
        and not unknown_groups
        and freshness_audit["schema_passed"].all()
        and freshness_audit["freshness_passed"].all()
    )
    strict_operational_gate = bool(
        operational_core_gate
        and point_in_time_vintage_gate
        and release_timing_gate
        and investable_return_source_gate
        and economic_review_approved_gate
        and publication_lag_review_gate
        and duplicate_information_review_gate
    )
    waiver_registry = pd.read_csv(OPERATIONAL_RISK_ACCEPTANCE_FILE)
    expected_waivers = {
        "alfred_vintage_missing",
        "investable_return_missing",
        "human_economic_review_missing",
    }
    if waiver_registry["risk_id"].duplicated().any():
        raise ValueError("Operational risk acceptance contains duplicate risk_id")
    missing_waivers = expected_waivers.difference(waiver_registry["risk_id"])
    if missing_waivers:
        raise ValueError(f"Operational risk acceptance missing: {sorted(missing_waivers)}")
    waiver_registry["accepted"] = waiver_registry["accepted"].map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes", "y"}
    )
    waiver_rows = waiver_registry.loc[
        waiver_registry["risk_id"].isin(expected_waivers)
    ].copy()
    risk_is_resolved = {
        "alfred_vintage_missing": bool(
            point_in_time_vintage_gate and release_timing_gate
        ),
        "investable_return_missing": investable_return_source_gate,
        "human_economic_review_missing": bool(
            economic_review_approved_gate
            and publication_lag_review_gate
            and duplicate_information_review_gate
        ),
    }
    waiver_rows["risk_currently_present"] = waiver_rows["risk_id"].map(
        lambda risk_id: not risk_is_resolved[risk_id]
    )
    waiver_rows["applied_to_current_run"] = (
        waiver_rows["risk_currently_present"]
        & waiver_rows["accepted"]
        & waiver_rows["scope"].eq("operational_gate_only")
        & waiver_rows["authority_source"].notna()
    )
    active_waiver_rows = waiver_rows.loc[waiver_rows["risk_currently_present"]]
    waiver_gate = bool(
        active_waiver_rows["accepted"].all()
        and active_waiver_rows["scope"].eq("operational_gate_only").all()
        and active_waiver_rows["authority_source"].notna().all()
    )
    waiver_applied = waiver_rows.set_index("risk_id")[
        "applied_to_current_run"
    ].to_dict()
    operational_gate = bool(
        operational_core_gate
        and (
            strict_operational_gate
            or (OPERATIONAL_GATE_PROFILE == "research_waiver" and waiver_gate)
        )
    )
    waiver_rows.assign(
        profile=OPERATIONAL_GATE_PROFILE,
        strict_gate_passed=strict_operational_gate,
        waiver_gate_passed=waiver_gate,
        operational_gate_passed=operational_gate,
    ).to_csv(RESULT_DIR / "operational_risk_acceptance_audit.csv", index=False)
    # A research waiver can permit continued research operations, never live
    # deployment.  Strict deployment and forward-shadow eligibility are kept
    # as separate decisions so an operational waiver cannot become a loophole.
    forward_shadow_eligible = bool(signal_gate and portfolio_gate and operational_gate)
    deployment_eligible = bool(
        signal_gate and portfolio_gate and strict_operational_gate
    )
    deployment_status = (
        "strict deployment eligible"
        if deployment_eligible
        else "forward shadow research eligible; strict deployment blocked"
        if forward_shadow_eligible
        else "research only: one or more deployment gates failed"
    )

    gate_detail_rows = [
        {
            "domain": "signal",
            "check": "aggregate_auc_above_0.5",
            "actual_value": signal_gate_summary["aggregate_auc"],
            "required_value": ">0.5",
            "passed": signal_gate_summary["aggregate_auc_passed"],
            "blocks_strict_deployment": True,
            "waived_for_research_operation": False,
            "evidence_file": "signal_gate_diagnostics.csv",
            "remediation": "improve pre-holdout nested OOS discrimination without holdout tuning",
        },
        {
            "domain": "signal",
            "check": "aggregate_rank_ic_above_0",
            "actual_value": signal_gate_summary["aggregate_rank_ic"],
            "required_value": ">0",
            "passed": signal_gate_summary["aggregate_rank_ic_passed"],
            "blocks_strict_deployment": True,
            "waived_for_research_operation": False,
            "evidence_file": "signal_gate_diagnostics.csv",
            "remediation": "inspect fold IC and coefficient diagnostics; do not flip using holdout",
        },
        {
            "domain": "signal",
            "check": "all_eligible_folds_evaluable",
            "actual_value": signal_gate_summary["fold_evaluable_ratio"],
            "required_value": "1.0",
            "passed": signal_gate_summary["fold_coverage_passed"],
            "blocks_strict_deployment": True,
            "waived_for_research_operation": False,
            "evidence_file": "outer_fold_signal_metrics.csv",
            "remediation": "remove late/missing fold inputs using predeclared data rules",
        },
        {
            "domain": "signal",
            "check": "fold_joint_auc_ic_pass_ratio",
            "actual_value": fold_signal_pass_ratio,
            "required_value": f">={2 / 3}",
            "passed": signal_gate_summary["fold_direction_passed"],
            "blocks_strict_deployment": True,
            "waived_for_research_operation": False,
            "evidence_file": "outer_fold_signal_metrics.csv",
            "remediation": "review temporal instability and recurring coefficient signs pre-holdout",
        },
        {
            "domain": "portfolio",
            "check": "median_fold_sharpe_gain",
            "actual_value": selected_policy_decision[
                "median_fold_Sharpe_difference"
            ],
            "required_value": ">=0.10",
            "passed": selected_policy_decision["median_fold_Sharpe_gate"],
            "blocks_strict_deployment": True,
            "waived_for_research_operation": False,
            "evidence_file": "position_sizing_policy_gate.csv",
            "remediation": "improve nested pre-holdout timing alpha versus same exposure",
        },
        {
            "domain": "portfolio",
            "check": "positive_active_return_fold_ratio",
            "actual_value": selected_policy_decision["positive_fold_ratio"],
            "required_value": f">={2 / 3}",
            "passed": selected_policy_decision["positive_fold_ratio_gate"],
            "blocks_strict_deployment": True,
            "waived_for_research_operation": False,
            "evidence_file": "position_sizing_policy_gate.csv",
            "remediation": "require active return to persist across nested folds",
        },
        {
            "domain": "portfolio",
            "check": "active_return_positive_at_10_and_25_bps",
            "actual_value": (
                f"10bp={selected_policy_decision['annualized_active_return_10bps']};"
                f"25bp={selected_policy_decision['annualized_active_return_25bps']}"
            ),
            "required_value": "both >0",
            "passed": selected_policy_decision["cost_10_25_positive"],
            "blocks_strict_deployment": True,
            "waived_for_research_operation": False,
            "evidence_file": "position_sizing_policy_gate.csv",
            "remediation": "demonstrate positive active return after realistic costs",
        },
        {
            "domain": "portfolio",
            "check": "drawdown_difference_floor",
            "actual_value": (
                f"10bp={selected_policy_decision['drawdown_difference_10bps']};"
                f"25bp={selected_policy_decision['drawdown_difference_25bps']}"
            ),
            "required_value": "both >=-0.03",
            "passed": selected_policy_decision["drawdown_gate"],
            "blocks_strict_deployment": True,
            "waived_for_research_operation": False,
            "evidence_file": "position_sizing_policy_gate.csv",
            "remediation": "avoid worsening drawdown by more than 3 percentage points",
        },
        {
            "domain": "operational",
            "check": "selected_sources_real_time_vintage_safe",
            "actual_value": point_in_time_vintage_gate,
            "required_value": "True",
            "passed": point_in_time_vintage_gate,
            "blocks_strict_deployment": True,
            "waived_for_research_operation": waiver_applied[
                "alfred_vintage_missing"
            ],
            "evidence_file": "selected_source_point_in_time_audit.csv",
            "remediation": "replace selected FRED histories with ALFRED real-time vintages",
        },
        {
            "domain": "operational",
            "check": "selected_sources_release_timing_safe",
            "actual_value": release_timing_gate,
            "required_value": "True",
            "passed": release_timing_gate,
            "blocks_strict_deployment": True,
            "waived_for_research_operation": waiver_applied[
                "alfred_vintage_missing"
            ],
            "evidence_file": "selected_source_point_in_time_audit.csv",
            "remediation": "verify month-end availability and next-month execution for every source",
        },
        {
            "domain": "operational",
            "check": "investable_distribution_adjusted_return_source",
            "actual_value": investable_return_source_gate,
            "required_value": "True",
            "passed": investable_return_source_gate,
            "blocks_strict_deployment": True,
            "waived_for_research_operation": waiver_applied[
                "investable_return_missing"
            ],
            "evidence_file": "market_return_role_registry.csv",
            "remediation": (
                f"provide audited {market_profile.display_name} total-return "
                "index or ETF adjusted-price CSV"
            ),
        },
        {
            "domain": "operational",
            "check": "human_economic_review_approved",
            "actual_value": economic_review_approved_gate,
            "required_value": "True",
            "passed": economic_review_approved_gate,
            "blocks_strict_deployment": True,
            "waived_for_research_operation": waiver_applied[
                "human_economic_review_missing"
            ],
            "evidence_file": "economic_review_checklist.csv",
            "remediation": "complete and approve economic_review_registry.csv",
        },
        {
            "domain": "operational",
            "check": "publication_lag_human_review",
            "actual_value": publication_lag_review_gate,
            "required_value": "True",
            "passed": publication_lag_review_gate,
            "blocks_strict_deployment": True,
            "waived_for_research_operation": waiver_applied[
                "human_economic_review_missing"
            ],
            "evidence_file": "economic_review_checklist.csv",
            "remediation": "review selected sources against actual publication calendars",
        },
        {
            "domain": "operational",
            "check": "duplicate_information_human_review",
            "actual_value": duplicate_information_review_gate,
            "required_value": "True",
            "passed": duplicate_information_review_gate,
            "blocks_strict_deployment": True,
            "waived_for_research_operation": waiver_applied[
                "human_economic_review_missing"
            ],
            "evidence_file": "economic_review_checklist.csv",
            "remediation": "approve economic overlap review for selected factors",
        },
    ]
    gate_details = pd.DataFrame(gate_detail_rows)
    gate_details.to_csv(RESULT_DIR / "deployment_gate_details.csv", index=False)
    gate_details.loc[
        gate_details["blocks_strict_deployment"] & ~gate_details["passed"]
    ].to_csv(RESULT_DIR / "deployment_blockers.csv", index=False)
    pd.DataFrame(
        [
            {
                "signal_gate": signal_gate,
                "portfolio_gate": portfolio_gate,
                "historical_holdout_safety_gate": primary_holdout_safety_gate,
                "primary_target_mode": active_primary_target_spec["mode"],
                "primary_target_spec_json": json.dumps(
                    active_primary_target_spec, sort_keys=True
                ),
                "operational_gate": operational_gate,
                "operational_core_gate": operational_core_gate,
                "strict_operational_gate": strict_operational_gate,
                "waiver_gate": waiver_gate,
                "operational_gate_profile": OPERATIONAL_GATE_PROFILE,
                "waived_risks": "|".join(
                    sorted(
                        risk_id
                        for risk_id, applied in waiver_applied.items()
                        if applied
                    )
                ),
                # Deprecated compatibility alias.  This is a fold joint-pass
                # ratio, not coefficient-sign stability.
                "direction_stability": fold_signal_pass_ratio,
                "direction_stability_definition": signal_gate_summary[
                    "direction_metric_definition"
                ],
                "fold_joint_direction_pass_ratio": fold_signal_pass_ratio,
                "fold_evaluable_ratio": signal_gate_summary[
                    "fold_evaluable_ratio"
                ],
                "coefficient_sign_consistency_min_diagnostic": (
                    coefficient_sign_consistency_min
                ),
                "coefficient_sign_consistency_median_diagnostic": (
                    coefficient_sign_consistency_median
                ),
                "coefficient_sign_consistency_definition": (
                    "dominant standardized Logistic coefficient sign share by required "
                    "family across outer folds; diagnostic, not a deployment gate"
                ),
                "economic_review_status": (
                    "approved"
                    if economic_review_approved_gate
                    and publication_lag_review_gate
                    and duplicate_information_review_gate
                    else "pending_but_waived_for_research_operation"
                    if waiver_applied["human_economic_review_missing"]
                    else "pending"
                ),
                "point_in_time_vintage_gate": point_in_time_vintage_gate,
                "release_timing_gate": release_timing_gate,
                "data_freshness_schema_gate": bool(
                    freshness_audit["schema_passed"].all()
                    and freshness_audit["freshness_passed"].all()
                ),
                "investable_return_source_gate": investable_return_source_gate,
                "economic_review_approved_gate": economic_review_approved_gate,
                "publication_lag_review_gate": publication_lag_review_gate,
                "duplicate_information_review_gate": duplicate_information_review_gate,
                "accuracy_lift_vs_naive_diagnostic_only": final_metrics[
                    "accuracy_lift_vs_naive"
                ],
                "deployment_eligible": deployment_eligible,
                "forward_shadow_eligible": forward_shadow_eligible,
                "status": deployment_status,
            }
        ]
    ).to_csv(RESULT_DIR / "deployment_gates.csv", index=False)

    # ⑧ CURRENT EWS (configured final model only)
    print()
    print(f"⑧ 현재 {final_label} EWS 계산")
    latest_kwargs = _model_runtime_kwargs(
        FINAL_MODEL_TYPE, mlp_params=active_mlp_params
    )
    latest_min_train = _model_min_train_months(FINAL_MODEL_TYPE)

    latest = fit_latest_ews(
        X=X,
        y=y,
        features=selected_features,
        horizon=FORECAST_HORIZON,
        asof_date=portfolio_prediction_end,
        min_train=latest_min_train,
        model_type=FINAL_MODEL_TYPE,
        market_name=MARKET_NAME,
        **latest_kwargs,
    )
    current_ews = latest["ews"]
    current_state = ews_state(
        current_ews,
        risk_off=EWS_RISK_OFF,
        risk_on=EWS_RISK_ON,
    )
    current_score_history = pd.concat(
        [
            nested_prediction * 100,
            portfolio_predictions[FINAL_MODEL_TYPE] * 100,
            pd.Series({latest["date"]: current_ews}),
        ]
    ).sort_index()
    current_score_history = current_score_history[
        ~current_score_history.index.duplicated(keep="last")
    ]
    current_target_history = target_weight_from_ews(
        current_score_history,
        policy=selected_policy,
        **sizing_config,
    )
    current_stock_weight = float(current_target_history.loc[latest["date"]])
    current_executed_weight = float(
        current_target_history.shift(1).loc[latest["date"]]
    )
    current_cash_weight = 1.0 - current_stock_weight

    if "feature_values" in latest:
        latest["feature_values"].to_csv(
            RESULT_DIR / "latest_feature_values.csv"
        )

    latest_result = {
        "date": str(latest["date"].date()),
        "market_key": market_profile.key,
        "market_name": market_profile.display_name,
        "market_ticker": market_profile.ticker,
        "investable_instrument": market_profile.investable_instrument,
        "investable_ticker": market_profile.investable_ticker,
        "portfolio_benchmark_instrument": portfolio_instrument,
        "portfolio_benchmark_ticker": portfolio_ticker,
        "model": final_label,
        "target_mode": active_primary_target_spec["mode"],
        "target_spec": active_primary_target_spec,
        "raw_ews": float(current_ews),
        "ews": float(current_ews),
        "signal_state": current_state,
        "deployment_eligible": deployment_eligible,
        "forward_shadow_eligible": forward_shadow_eligible,
        "deployment_status": deployment_status,
        "allocation_policy": selected_policy,
        "tactical_overlay_active": not allocation_fallback_used,
        "fail_closed_fallback_used": allocation_fallback_used,
        "target_stock_weight": current_stock_weight,
        "executed_stock_weight": current_executed_weight,
        "stock_weight": current_stock_weight,
        "cash_weight": current_cash_weight,
        "training_observations": int(latest["train_n"]),
        "features": selected_features,
        "test_auc": _json_number(final_metrics["auc"]),
        "test_accuracy": _json_number(final_metrics["accuracy"]),
        "naive_accuracy": _json_number(final_metrics["naive_accuracy"]),
        "rank_ic": _json_number(final_ic["rank_ic"]),
        "strategy_cagr": _json_number(final_strategy_row["CAGR"]),
        "strategy_sharpe": _json_number(final_strategy_row["Sharpe"]),
        "strategy_max_drawdown": _json_number(
            final_strategy_row["MaxDrawdown"]
        ),
        "performance_start": final_strategy_row["Start"],
        "performance_end": final_strategy_row["End"],
        "performance_months": int(final_strategy_row["Months"]),
    }
    with open(
        RESULT_DIR / "latest_ews.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(latest_result, fp, ensure_ascii=False, indent=2)
    if FINAL_MODEL_TYPE == "svm":
        with open(
            RESULT_DIR / "latest_svm_ews.json",
            "w",
            encoding="utf-8",
        ) as fp:
            json.dump(latest_result, fp, ensure_ascii=False, indent=2)

    forward_shadow_spec = {
        "market_key": market_profile.key,
        "market_name": market_profile.display_name,
        "market_ticker": market_profile.ticker,
        "investable_instrument": market_profile.investable_instrument,
        "investable_ticker": market_profile.investable_ticker,
        "portfolio_benchmark_instrument": portfolio_instrument,
        "portfolio_benchmark_ticker": portfolio_ticker,
        "status": (
            "ready_for_next_observation"
            if forward_shadow_eligible
            else "blocked_until_all_deployment_gates_pass"
        ),
        "strict_deployment_eligible": deployment_eligible,
        "research_waiver_may_not_authorize_live_deployment": True,
        "freeze_date": str(latest["date"].date()),
        "first_eligible_observation": str(
            (latest["date"] + pd.offsets.MonthEnd(1)).date()
        ),
        "minimum_shadow_months": 12,
        "historical_holdout_may_not_change_spec": True,
        "model": FINAL_MODEL_TYPE,
        "model_target_mode": active_primary_target_spec["mode"],
        "model_target_spec": active_primary_target_spec,
        "features": selected_features,
        "allocation_policy": selected_policy,
        "tactical_overlay_active": not allocation_fallback_used,
        "fail_closed_fallback_used": allocation_fallback_used,
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "monitoring": {
            "missing_frozen_feature": "stop",
            "raw_ews_outside_0_100": "stop",
            "target_weight_outside_limits": "stop",
            "score_population_stability_index_warning": 0.25,
            "monthly_turnover_warning": 0.40,
            "active_drawdown_vs_same_exposure_stop": -0.10,
            "calibration_window_months": 24,
            "calibration_slope_warning_range": [0.50, 1.50],
        },
    }
    forward_shadow_spec["freeze_hash"] = canonical_spec_hash(forward_shadow_spec)
    with open(
        RESULT_DIR / "forward_shadow_spec.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(forward_shadow_spec, fp, ensure_ascii=False, indent=2)
    initialize_shadow_ledger(RESULT_DIR / "forward_shadow_ledger.csv")

    # Freeze the independently validated MLP candidate as a paper/shadow
    # strategy.  Failed validation gates can never authorize capital, but the
    # append-only shadow ledger lets genuinely new observations accumulate.
    mlp_latest = fit_latest_ews(
        X=X,
        y=mlp_y,
        features=model_feature_sets["mlp"],
        horizon=FORECAST_HORIZON,
        asof_date=portfolio_prediction_end,
        min_train=_model_min_train_months("mlp"),
        model_type="mlp",
        market_name=MARKET_NAME,
        **_model_runtime_kwargs("mlp", mlp_params=active_mlp_params),
    )
    mlp_current_ews = float(mlp_latest["ews"])
    mlp_current_score_history = pd.concat(
        [
            mlp_nested_prediction * 100,
            portfolio_predictions["mlp"] * 100,
            pd.Series({mlp_latest["date"]: mlp_current_ews}),
        ]
    ).sort_index()
    mlp_current_score_history = mlp_current_score_history[
        ~mlp_current_score_history.index.duplicated(keep="last")
    ]
    mlp_current_target_history = target_weight_from_ews(
        mlp_current_score_history,
        policy=mlp_selected_policy,
        **sizing_config,
    )
    mlp_executed_weight_history = mlp_current_target_history.shift(1)
    mlp_turnover_history = mlp_executed_weight_history.diff().abs()
    mlp_score_history_path = RESULT_DIR / "mlp_frozen_score_history.csv"
    pd.DataFrame(
        {
            "raw_ews": mlp_current_score_history,
            "target_stock_weight": mlp_current_target_history,
            "executed_stock_weight": mlp_executed_weight_history,
            "turnover": mlp_turnover_history,
        }
    ).to_csv(mlp_score_history_path, index_label="observation_date")
    mlp_current_stock_weight = float(
        mlp_current_target_history.loc[mlp_latest["date"]]
    )
    mlp_current_executed_weight = float(
        mlp_executed_weight_history.loc[mlp_latest["date"]]
    )
    mlp_current_turnover = float(
        mlp_turnover_history.loc[mlp_latest["date"]]
    )
    mlp_holdout_metrics = model_comparison.loc[
        model_comparison["model"].eq("MLP")
    ].iloc[0]
    mlp_holdout_ic = ic_comparison.loc[ic_comparison["model"].eq("MLP")].iloc[0]
    mlp_holdout_strategy = performance.loc[
        (performance["model"].eq("MLP"))
        & (performance["strategy"].eq("MLP Dynamic"))
    ].iloc[0]
    mlp_latest_result = {
        "date": str(mlp_latest["date"].date()),
        "market_key": market_profile.key,
        "market_name": market_profile.display_name,
        "market_ticker": market_profile.ticker,
        "investable_instrument": market_profile.investable_instrument,
        "investable_ticker": market_profile.investable_ticker,
        "portfolio_benchmark_instrument": portfolio_instrument,
        "portfolio_benchmark_ticker": portfolio_ticker,
        "model": "MLP",
        "target_mode": active_mlp_target_spec["mode"],
        "target_spec": active_mlp_target_spec,
        "usage_mode": (
            "capital_and_shadow" if mlp_capital_use_allowed else "research_shadow_only"
        ),
        "raw_ews": mlp_current_ews,
        "ews": mlp_current_ews,
        "signal_state": ews_state(
            mlp_current_ews,
            risk_off=EWS_RISK_OFF,
            risk_on=EWS_RISK_ON,
        ),
        "capital_use_allowed": mlp_capital_use_allowed,
        "research_shadow_allowed": True,
        "validation_gates": mlp_validation_row,
        "allocation_policy": mlp_selected_policy,
        "tactical_overlay_active": not mlp_allocation_fallback_used,
        "fail_closed_fallback_used": mlp_allocation_fallback_used,
        "target_stock_weight": mlp_current_stock_weight,
        "executed_stock_weight": mlp_current_executed_weight,
        "turnover": mlp_current_turnover,
        "cash_weight": 1.0 - mlp_current_stock_weight,
        "training_observations": int(mlp_latest["train_n"]),
        "features": model_feature_sets["mlp"],
        "historical_holdout_diagnostic": {
            "auc": _json_number(mlp_holdout_metrics["auc"]),
            "rank_ic": _json_number(mlp_holdout_ic["rank_ic"]),
            "strategy_cagr": _json_number(mlp_holdout_strategy["CAGR"]),
            "strategy_sharpe": _json_number(mlp_holdout_strategy["Sharpe"]),
            "strategy_max_drawdown": _json_number(
                mlp_holdout_strategy["MaxDrawdown"]
            ),
            "evaluated_locked_allocation_policy": mlp_pre_veto_selected_policy,
            "used_for_positive_promotion": False,
            "used_as_fail_closed_safety_veto": True,
            "safety_gate": mlp_holdout_safety_gate,
            "applied_allocation_policy": mlp_selected_policy,
        },
    }
    with open(
        RESULT_DIR / "latest_mlp_ews.json", "w", encoding="utf-8"
    ) as fp:
        json.dump(mlp_latest_result, fp, ensure_ascii=False, indent=2)
    mlp_same_exposure_row = mlp_pre_comparison.loc[
        mlp_pre_comparison["policy"].eq(mlp_selected_policy)
        & mlp_pre_comparison["transaction_cost_bps"].eq(TRANSACTION_COST_BPS)
    ]
    if len(mlp_same_exposure_row) != 1:
        raise RuntimeError("MLP frozen same-exposure benchmark is ambiguous")
    mlp_same_exposure_stock_weight = float(
        mlp_same_exposure_row.iloc[0]["average_stock_weight"]
    )
    mlp_shadow_spec = {
        "schema_version": 1,
        "status": (
            "ready_for_next_observation"
            if mlp_capital_use_allowed
            else "research_shadow_only"
        ),
        "capital_authorized": mlp_capital_use_allowed,
        "research_shadow_authorized": True,
        "strict_deployment_eligible": mlp_capital_use_allowed,
        "freeze_date": str(mlp_latest["date"].date()),
        "first_eligible_observation": str(
            (mlp_latest["date"] + pd.offsets.MonthEnd(1)).date()
        ),
        "minimum_shadow_months": 12,
        "historical_holdout_may_not_change_spec": True,
        "model": "mlp",
        "model_target_mode": active_mlp_target_spec["mode"],
        "model_target_spec": active_mlp_target_spec,
        "features": model_feature_sets["mlp"],
        "scoring_protocol": "monthly_expanding_refit_v1",
        "forecast_horizon_months": FORECAST_HORIZON,
        "minimum_training_months": _model_min_train_months("mlp"),
        "refit_every_months": MLP_REFIT_EVERY,
        "random_state": RANDOM_SEED,
        "model_params": active_mlp_params,
        "allocation_policy": mlp_selected_policy,
        "sizing_config": sizing_config,
        "stock_weight_limits": [MIN_STOCK_WEIGHT, MAX_STOCK_WEIGHT],
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "cash_return_convention": "simple_divide_12",
        "same_exposure_stock_weight": mlp_same_exposure_stock_weight,
        "factor_matrix_file": "factor_matrix.parquet",
        "target_file": "mlp_target.csv",
        "market_key": market_profile.key,
        "market_name": market_profile.display_name,
        "market_ticker": market_profile.ticker,
        "investable_instrument": market_profile.investable_instrument,
        "investable_ticker": market_profile.investable_ticker,
        "portfolio_benchmark_instrument": portfolio_instrument,
        "portfolio_benchmark_ticker": portfolio_ticker,
        "signal_market_file": "market_monthly.csv",
        "signal_market_column": MARKET_SERIES_NAME,
        "portfolio_price_file": "portfolio_return_source_monthly.csv",
        "portfolio_price_column": f"investable_{MARKET_SERIES_NAME}",
        "cash_yield_file": "monthly_panel.parquet",
        "cash_yield_column": "cash_yield_3m",
        "score_history_file": mlp_score_history_path.name,
        "score_history_sha256": sha256_file(mlp_score_history_path),
        "validation_evidence": "mlp_validation_gates.csv",
        "ledger_file": "mlp_research_shadow_ledger.csv",
        "monitoring": {
            "missing_frozen_feature": "stop",
            "raw_ews_outside_0_100": "stop",
            "target_weight_outside_limits": "stop",
            "score_population_stability_index_warning": 0.25,
            "monthly_turnover_warning": 0.40,
            "active_drawdown_vs_same_exposure_stop": -0.10,
            "calibration_window_months": 24,
            "calibration_slope_warning_range": [0.50, 1.50],
            "required_realized_fields": [
                "strategy_return",
                "same_exposure_return",
                "active_return",
            ],
        },
    }
    mlp_shadow_spec["freeze_hash"] = canonical_spec_hash(mlp_shadow_spec)
    with open(
        RESULT_DIR / "mlp_research_shadow_spec.json", "w", encoding="utf-8"
    ) as fp:
        json.dump(mlp_shadow_spec, fp, ensure_ascii=False, indent=2)
    initialize_shadow_ledger(RESULT_DIR / "mlp_research_shadow_ledger.csv")
    mlp_shadow_observation_template = {
        "observation_date": mlp_shadow_spec["first_eligible_observation"],
        "freeze_hash": mlp_shadow_spec["freeze_hash"],
        "feature_values": {
            feature: None for feature in model_feature_sets["mlp"]
        },
        "raw_ews": None,
        "target_stock_weight": None,
        "executed_stock_weight": None,
        "turnover": None,
        "strategy_return": None,
        "same_exposure_return": None,
        "active_return": None,
        "score_psi": None,
        "calibration_slope": None,
    }
    with open(
        RESULT_DIR / "mlp_shadow_observation_template.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            mlp_shadow_observation_template,
            fp,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 70)
    print(f"[CURRENT] CURRENT {final_label.upper()} EWS")
    print("=" * 70)
    print(f"기준일       : {latest['date'].date()}")
    print(f"EWS          : {current_ews:.1f} / 100")
    print(f"모델 신호    : {current_state}")
    print(f"실전 상태    : {deployment_status}")
    print(f"배분 정책    : {selected_policy}")
    print(
        "타이밍 오버레이: "
        + ("활성" if not allocation_fallback_used else "비활성 (gate 실패·정적 fallback)")
    )
    print(f"{MARKET_NAME} 목표 비중: {current_stock_weight:.1%}")
    print(f"현금 비중    : {current_cash_weight:.1%}")

    # ⑨ VISUALIZATION
    plot_dashboard(
        backtest=final_backtest,
        future_return=target["future_return"],
        rolling_ic=final_rolling_ic,
        rolling_sharpe=rolling_sr,
        save_path=RESULT_DIR / "ews_dashboard.png",
        market_name=MARKET_NAME,
    )
    plot_latest_allocation(
        stock_weight=current_stock_weight,
        cash_weight=current_cash_weight,
        ews=current_ews,
        date=latest["date"].date(),
        save_path=RESULT_DIR / "current_allocation.png",
    )

    manifest.update(
        {
            "status": "complete",
            "selected_features": selected_features,
            "model_feature_sets": model_feature_sets,
            "original_reference_selected_features": original_selected_features,
            "selected_allocation_policy": selected_policy,
            "allocation_fallback_used": allocation_fallback_used,
            "research_period": {
                "start": target_index.min(),
                "end": pre_holdout_end,
            },
            "historical_research_holdout": {
                "start": split["test_start"],
                "end": portfolio_prediction_end,
                "labelled_signal_end": split["test_end"],
                "portfolio_performance_end": portfolio_prediction_end,
                "used_for_selection": False,
            },
            "deployment_gates": {
                "signal": signal_gate,
                "portfolio": portfolio_gate,
                "operational": operational_gate,
                "strict_operational": strict_operational_gate,
                "forward_shadow_eligible": forward_shadow_eligible,
                "strict_deployment_eligible": deployment_eligible,
            },
            "mlp_candidate_validation": {
                **mlp_validation_row,
                "shadow_spec": "mlp_research_shadow_spec.json",
                "shadow_ledger": "mlp_research_shadow_ledger.csv",
            },
            "output_files": sorted(
                path.name for path in RESULT_DIR.iterdir() if path.is_file()
            ),
        }
    )
    write_manifest(RESULT_DIR, manifest)

    print()
    print("[OK] 모든 결과 저장 완료 ->", RESULT_DIR)


if __name__ == "__main__":
    arguments = parse_args()
    main(
        run_id=arguments.run_id,
        quick=arguments.quick,
        allow_partial_raw_universe=arguments.allow_partial_raw_universe,
        market_key=arguments.market,
        market_file=arguments.market_file,
        market_metadata_file=arguments.market_metadata_file,
        investable_market_file=arguments.investable_market_file,
    )
