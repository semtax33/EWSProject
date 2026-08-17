import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np

from src.validation import (
    OuterFold,
    coefficient_family_sign_stability,
    enforce_signal_gate_fallback,
    compare_position_sizing,
    evaluate_holdout_safety_veto,
    evaluate_signal_gate,
    fixed_fold_availability_audit,
    fixed_outer_predict,
    fold_candidate_availability_audit,
    logistic_fold_coefficient_audit,
    make_purged_outer_folds,
    nested_outer_predict,
    screening_evaluation_start,
    select_position_policy,
)


class OuterFoldTests(unittest.TestCase):
    def test_holdout_safety_can_only_pass_or_veto_locked_policy(self):
        passed, checks, differences = evaluate_holdout_safety_veto(
            auc=0.61,
            rank_ic=0.08,
            dynamic_sharpe=1.20,
            same_exposure_sharpe=1.00,
            dynamic_cagr=0.14,
            same_exposure_cagr=0.12,
            dynamic_max_drawdown=-0.10,
            same_exposure_max_drawdown=-0.12,
        )
        self.assertTrue(passed)
        self.assertTrue(all(checks.values()))
        self.assertGreater(differences["sharpe_difference"], 0)

        vetoed, failed, _ = evaluate_holdout_safety_veto(
            auc=0.56,
            rank_ic=0.01,
            dynamic_sharpe=0.67,
            same_exposure_sharpe=1.03,
            dynamic_cagr=0.11,
            same_exposure_cagr=0.14,
            dynamic_max_drawdown=-0.22,
            same_exposure_max_drawdown=-0.16,
        )
        self.assertFalse(vetoed)
        self.assertFalse(failed["sharpe_not_worse"])
        self.assertFalse(failed["cagr_not_worse"])
        self.assertFalse(failed["drawdown_not_worse_by_more_than_3pct"])

    def test_required_family_coefficient_stability_spans_transform_changes(self):
        audit = pd.DataFrame(
            {
                "feature": ["term__chg_3m", "term__z_24m", "turn__chg_6m"],
                "base": ["term", "term", "turn"],
                "fit_status": ["fitted", "fitted", "fitted"],
                "coefficient_sign": ["negative", "negative", "positive"],
            }
        )
        summary = coefficient_family_sign_stability(
            audit,
            {"term_spread": ("term",), "turnover_trend": ("turn",)},
        ).set_index("required_family")
        self.assertEqual(summary.loc["term_spread", "fold_occurrences"], 2)
        self.assertEqual(summary.loc["term_spread", "distinct_transforms"], 2)
        self.assertEqual(summary.loc["term_spread", "sign_consistency_ratio"], 1.0)
        self.assertFalse(
            summary.loc["turnover_trend", "enough_recurrence_to_assess"]
        )

    def test_fold_windows_are_purged_and_end_before_holdout(self):
        index = pd.date_range("1990-01-31", "2025-12-31", freq="ME")
        folds = make_purged_outer_folds(
            index,
            research_end="2019-12-31",
            min_train_months=60,
            inner_validation_months=24,
            outer_validation_months=24,
            purge_months=3,
            screening_oos_months=36,
        )
        self.assertGreater(len(folds), 1)
        for fold in folds:
            inner_gap = (
                fold.outer_start.to_period("M")
                - fold.inner_validation_end.to_period("M")
            ).n
            development_gap = (
                fold.inner_validation_start.to_period("M")
                - fold.development_end.to_period("M")
            ).n
            self.assertGreaterEqual(inner_gap, 4)
            self.assertGreaterEqual(development_gap, 4)
            self.assertLessEqual(fold.outer_end, pd.Timestamp("2019-12-31"))

        first = folds[0]
        development_index = index[index <= first.development_end]
        # 60 train + 3-month purge + 36 OOS, with one shared boundary row.
        self.assertGreaterEqual(len(development_index), 60 + 3 + 36 - 1)
        screen_start = screening_evaluation_start(
            index,
            development_end=first.development_end,
            min_train_months=60,
            purge_months=3,
            min_oos_predictions=36,
        )
        screen_position = development_index.get_loc(screen_start)
        self.assertGreaterEqual(screen_position - 3 + 1, 60)
        self.assertGreaterEqual(len(development_index) - screen_position, 36)

    def test_screening_start_rejects_old_under_sized_first_fold(self):
        index = pd.date_range("1996-03-31", periods=96, freq="ME")
        with self.assertRaisesRegex(ValueError, "available=34, required=36"):
            screening_evaluation_start(
                index,
                development_end=index[-1],
                min_train_months=60,
                purge_months=3,
                min_oos_predictions=36,
            )

    def test_signal_gate_reports_coverage_direction_and_short_fold_separately(self):
        folds = pd.DataFrame(
            {
                "fold": [1, 2, 3],
                "expected_observations": [24, 24, 12],
                "observations": [24, 10, 12],
                "auc": [0.60, np.nan, np.nan],
                "rank_ic": [0.10, 0.05, np.nan],
            }
        )
        summary, details = evaluate_signal_gate(
            folds,
            aggregate_auc=0.61,
            aggregate_rank_ic=0.08,
        )
        self.assertEqual(summary["gate_eligible_folds"], 2)
        self.assertEqual(summary["fold_evaluable_ratio"], 0.5)
        self.assertEqual(summary["fold_joint_direction_pass_ratio"], 0.5)
        self.assertFalse(summary["signal_gate_passed"])
        self.assertIn("not factor-coefficient", summary["direction_metric_definition"])
        self.assertIn("insufficient_predictions", details.iloc[1]["failure_reason"])
        self.assertIn("auc_unavailable", details.iloc[1]["failure_reason"])
        self.assertNotIn("rank_ic_not_above_0", details.iloc[1]["failure_reason"])
        self.assertTrue(details.iloc[1]["rank_ic_threshold_passed"])
        self.assertFalse(details.iloc[1]["rank_ic_passed"])
        self.assertFalse(details.iloc[2]["gate_eligible_fold"])

    def test_pre_model_inception_fold_is_declared_and_excluded(self):
        folds = pd.DataFrame(
            {
                "fold": [1, 2, 3],
                "model_eligibility_start": pd.to_datetime(
                    ["2004-01-31", "2004-01-31", "2004-01-31"]
                ),
                "expected_observations": [0, 24, 24],
                "observations": [0, 24, 24],
                "auc": [np.nan, 0.60, 0.62],
                "rank_ic": [np.nan, 0.10, 0.12],
            }
        )
        summary, details = evaluate_signal_gate(
            folds,
            aggregate_auc=0.61,
            aggregate_rank_ic=0.11,
        )

        self.assertEqual(summary["gate_eligible_folds"], 2)
        self.assertTrue(summary["signal_gate_passed"])
        self.assertEqual(
            details.iloc[0]["failure_reason"],
            "pre_model_inception;excluded_from_ratio",
        )

    def test_fixed_outer_set_cannot_change_between_folds(self):
        index = pd.date_range("1990-01-31", periods=150, freq="ME")
        X = pd.DataFrame(
            {
                "term__level": np.sin(np.arange(150) / 7),
                "trend": np.cos(np.arange(150) / 11),
            },
            index=index,
        )
        y = pd.Series((X["term__level"] > 0).astype(float), index=index)
        folds = [
            OuterFold(1, index[70], index[74], index[90], index[100], index[119]),
            OuterFold(2, index[90], index[94], index[110], index[120], index[139]),
        ]
        prediction, selections, screening = fixed_outer_predict(
            X=X,
            y=y,
            folds=folds,
            features=["term__level", "trend"],
            feature_groups={"term": "liquidity", "trend": "market_internal"},
            final_model_type="logistic",
            final_min_train_months=60,
            horizon=3,
        )

        self.assertEqual(prediction.notna().sum(), 40)
        self.assertTrue(screening.empty)
        selected_by_fold = selections.groupby("fold")["feature"].apply(tuple)
        self.assertEqual(selected_by_fold.loc[1], selected_by_fold.loc[2])
        self.assertTrue(
            selections["selection_note"].eq(
                "predeclared_structural_feature_set"
            ).all()
        )

    def test_fold_availability_excludes_only_untrainable_history(self):
        index = pd.date_range("2000-01-31", periods=60, freq="ME")
        X = pd.DataFrame(
            {"late__level": [np.nan] * 20 + list(np.arange(40.0))},
            index=index,
        )
        y = pd.Series(np.arange(60) % 2, index=index, dtype=float)
        folds = [
            OuterFold(1, index[25], index[27], index[30], index[33], index[38]),
            OuterFold(2, index[45], index[47], index[50], index[53], index[58]),
        ]
        audit = fold_candidate_availability_audit(
            X=X,
            y=y,
            folds=folds,
            min_train_months=12,
            horizon=2,
            min_oos_predictions=6,
            required_families={"late_family": ("late",)},
        ).set_index("fold")

        self.assertFalse(audit.loc[1, "fold_availability_passed"])
        self.assertTrue(audit.loc[2, "fold_availability_passed"])
        self.assertEqual(
            audit.loc[1, "exclusion_reason"],
            "pre_model_inception_or_insufficient_feature_history",
        )

    def test_fixed_fold_uses_outer_trainability_not_screening_history(self):
        index = pd.date_range("2000-01-31", periods=80, freq="ME")
        X = pd.DataFrame(
            {"fixed": [np.nan] * 10 + list(np.arange(70.0))}, index=index
        )
        y = pd.Series(np.arange(80) % 2, index=index, dtype=float)
        folds = [
            OuterFold(1, index[20], index[22], index[25], index[35], index[44]),
            OuterFold(2, index[40], index[42], index[45], index[55], index[64]),
        ]
        audit = fixed_fold_availability_audit(
            X=X,
            y=y,
            folds=folds,
            min_train_months=12,
            horizon=2,
        )
        self.assertTrue(audit["fold_availability_passed"].all())
        self.assertEqual(audit.loc[0, "evaluation_start"], index[35])

    def test_nested_mlp_can_audit_and_skip_only_unevaluable_fold(self):
        index = pd.date_range("2000-01-31", periods=180, freq="ME")
        names = [f"x{i}__level" for i in range(4)]
        X = pd.DataFrame(
            {name: np.sin(np.arange(180) / (i + 3)) for i, name in enumerate(names)},
            index=index,
        )
        y = pd.Series(np.arange(180) % 2, index=index, dtype=float)
        folds = [
            OuterFold(1, index[100], index[104], index[115], index[120], index[129]),
            OuterFold(2, index[120], index[124], index[135], index[140], index[149]),
        ]
        scores = pd.DataFrame(
            {
                "feature": names,
                "base": [f"x{i}" for i in range(4)],
                "n": [36] * 4,
                "auc": [0.55] * 4,
                "brier": [0.24] * 4,
                "accuracy": [0.55] * 4,
                "auc_first": [0.54] * 4,
                "auc_second": [0.56] * 4,
                "stability_gap": [0.02] * 4,
                "rank_score": [0.54] * 4,
            }
        )
        history = pd.DataFrame([{"n": 0, "auc": np.nan, "rank_score": np.nan}])
        prediction = pd.Series(0.6, index=index[140:150])
        with (
            patch("src.validation.screen_single_factors", return_value=scores),
            patch("src.validation.prune_correlated_features", return_value=names),
            patch(
                "src.validation.exhaustive_combination_selection",
                side_effect=[([], history), (names, history)],
            ),
            patch("src.validation.walk_forward_predict", return_value=prediction),
        ):
            result, selections, screening = nested_outer_predict(
                X=X,
                y=y,
                folds=folds,
                feature_groups={f"x{i}": f"g{i}" for i in range(4)},
                screening_model_type="fast_logistic",
                selection_model_type="mlp",
                final_model_type="mlp",
                min_train_months=60,
                final_min_train_months=60,
                horizon=3,
                single_factor_refit_every=3,
                min_oos_predictions=36,
                top_feature_pool=10,
                raw_top_features_per_base=3,
                group_candidates_per_group=15,
                correlation_threshold=0.9,
                combination_candidate_pool=10,
                exhaustive_candidate_pool=6,
                min_model_features=4,
                max_model_features=6,
                min_validation_improvement=0.0,
                max_features_per_base=None,
                max_features_per_group=None,
                min_distinct_groups=1,
                svm_params={},
                mlp_params={"hidden_layer_sizes": (4,)},
                calibration_splits=3,
                allow_unavailable_folds=True,
            )
        self.assertEqual(set(selections["fold"]), {2})
        self.assertEqual(result.notna().sum(), 10)
        self.assertIn(
            "pre_model_inception_no_evaluable_combination",
            set(screening["selection_status"].dropna()),
        )

    def test_logistic_fold_coefficient_direction_is_audited(self):
        index = pd.date_range("2000-01-31", periods=110, freq="ME")
        x = np.sin(np.arange(len(index)) / 4)
        X = pd.DataFrame({"x__level": x}, index=index)
        y = pd.Series((x > 0).astype(float), index=index)
        fold = OuterFold(
            fold=1,
            development_end=index[60],
            inner_validation_start=index[64],
            inner_validation_end=index[75],
            outer_start=index[85],
            outer_end=index[100],
        )
        selections = pd.DataFrame(
            [{
                "fold": 1,
                "selection_rank": 1,
                "feature": "x__level",
                "base": "x",
                "group": "cycle",
            }]
        )
        audit = logistic_fold_coefficient_audit(
            X=X,
            y=y,
            folds=[fold],
            selections=selections,
            horizon=3,
            min_train_months=60,
        )
        self.assertEqual(audit.iloc[0]["coefficient_sign"], "positive")
        self.assertEqual(audit.iloc[0]["fit_status"], "fitted")
        self.assertIn("not causal", audit.iloc[0]["interpretation"])

    def test_logistic_fold_coefficient_aligns_short_target_index(self):
        full_index = pd.date_range("2000-01-31", periods=72, freq="ME")
        target_index = full_index[12:]
        X = pd.DataFrame(
            {"macro__level": np.linspace(-2, 2, len(full_index))},
            index=full_index,
        )
        y = pd.Series(
            (X.loc[target_index, "macro__level"] > 0).astype(int),
            index=target_index,
        )
        fold = OuterFold(
            fold=1,
            development_end=full_index[35],
            inner_validation_start=full_index[39],
            inner_validation_end=full_index[47],
            outer_start=full_index[60],
            outer_end=full_index[71],
        )
        selections = pd.DataFrame(
            [{
                "fold": 1,
                "selection_rank": 1,
                "feature": "macro__level",
                "base": "macro",
                "group": "growth",
            }]
        )

        audit = logistic_fold_coefficient_audit(
            X=X,
            y=y,
            folds=[fold],
            selections=selections,
            horizon=3,
            min_train_months=24,
        )

        self.assertEqual(audit.loc[0, "fit_status"], "fitted")
        self.assertLessEqual(audit.loc[0, "train_observations"], len(y))

    def test_portfolio_gate_exposes_each_failed_condition(self):
        comparison = pd.DataFrame(
            {
                "policy": ["linear", "linear"],
                "transaction_cost_bps": [10, 25],
                "annualized_active_return": [-0.01, -0.02],
                "drawdown_difference": [-0.01, -0.02],
            }
        )
        fold_results = pd.DataFrame(
            {
                "policy": ["linear"] * 5,
                "transaction_cost_bps": [10] * 5,
                "annualized_active_return": [0.01, -0.01, 0.02, -0.02, -0.01],
                "Sharpe_difference": [-0.1, -0.2, 0.0, 0.1, -0.05],
            }
        )
        policy, decision = select_position_policy(comparison, fold_results)
        row = decision.iloc[0]
        self.assertEqual(policy, "linear")
        self.assertFalse(row["portfolio_gate_passed"])
        self.assertTrue(row["drawdown_gate"])
        self.assertFalse(row["cost_10_25_positive"])
        self.assertIn("median_fold_sharpe", row["failed_conditions"])
        self.assertIn("active_return_not_positive", row["failed_conditions"])

    def test_failed_tactical_policies_select_declared_static_fallback(self):
        comparison = pd.DataFrame(
            {
                "policy": ["linear", "linear", "static_50_50", "static_50_50"],
                "transaction_cost_bps": [10, 25, 10, 25],
                "annualized_active_return": [-0.01, -0.02, 0.0, 0.0],
                "drawdown_difference": [-0.01, -0.02, 0.0, 0.0],
            }
        )
        fold_results = pd.DataFrame(
            {
                "policy": ["linear", "linear", "static_50_50", "static_50_50"],
                "transaction_cost_bps": [10, 10, 10, 10],
                "annualized_active_return": [-0.01, -0.02, 0.0, 0.0],
                "Sharpe_difference": [-0.1, -0.2, 0.0, 0.0],
            }
        )
        policy, decision = select_position_policy(
            comparison, fold_results, baseline="static_50_50"
        )
        self.assertEqual(policy, "static_50_50")
        selected = decision.loc[decision["selected"]].iloc[0]
        self.assertTrue(selected["selected_as_fallback"])
        self.assertIn("fail_closed", selected["selection_reason"])

    def test_passing_policy_selection_prioritizes_active_return_after_25bps(self):
        comparison = pd.DataFrame(
            {
                "policy": ["fixed_bin", "fixed_bin", "expanding_percentile", "expanding_percentile"],
                "transaction_cost_bps": [10, 25, 10, 25],
                "annualized_active_return": [0.020, 0.015, 0.024, 0.019],
                "drawdown_difference": [0.01, 0.01, 0.01, 0.01],
            }
        )
        fold_results = pd.DataFrame(
            {
                "policy": ["fixed_bin"] * 3 + ["expanding_percentile"] * 3,
                "transaction_cost_bps": [10] * 6,
                "annualized_active_return": [0.01] * 6,
                "Sharpe_difference": [0.30, 0.20, 0.10, 0.20, 0.15, 0.10],
            }
        )

        policy, decision = select_position_policy(
            comparison, fold_results, baseline="fixed_bin"
        )

        self.assertEqual(policy, "expanding_percentile")
        selected = decision.loc[decision["selected"]].iloc[0]
        self.assertIn("max_active_return_at_25bps", selected["selection_reason"])

    def test_failed_signal_gate_overrides_a_passing_tactical_policy(self):
        decisions = pd.DataFrame(
            {
                "policy": ["linear", "static_50_50"],
                "portfolio_gate_passed": [True, False],
                "selected": [True, False],
                "selected_as_fallback": [False, False],
                "selection_reason": ["pre_holdout_portfolio_gate_passed", "not_selected"],
            }
        )
        policy, result = enforce_signal_gate_fallback(
            "linear", decisions, signal_gate_passed=False
        )
        self.assertEqual(policy, "static_50_50")
        selected = result.loc[result["selected"]].iloc[0]
        self.assertTrue(selected["selected_as_fallback"])
        self.assertEqual(
            selected["selection_reason"],
            "signal_gate_failed;fail_closed_fallback",
        )

    def test_all_sizing_policies_use_identical_months(self):
        index = pd.date_range("2000-01-31", periods=100, freq="ME")
        market = pd.Series(100 * np.cumprod(np.repeat(1.01, 100)), index=index)
        score = pd.Series(np.linspace(20, 80, 100), index=index)
        comparison, monthly, _ = compare_position_sizing(
            market_price=market,
            raw_ews=score,
            cash_yield=None,
            policies=("linear", "fixed_bin", "expanding_percentile"),
            transaction_cost_scenarios=(10,),
            sizing_config={
                "min_weight": 0.2,
                "max_weight": 0.8,
                "fixed_thresholds": (35, 50, 65),
                "fixed_weights": (0.2, 0.4, 0.6, 0.8),
                "percentile_breaks": (0.2, 0.4, 0.6, 0.8),
                "percentile_weights": (0.2, 0.35, 0.5, 0.65, 0.8),
                "percentile_min_history": 36,
            },
        )
        self.assertEqual(comparison["Start"].nunique(), 1)
        self.assertEqual(comparison["End"].nunique(), 1)
        self.assertEqual(comparison["Months"].nunique(), 1)
        for _, group in monthly.groupby("policy"):
            self.assertEqual(len(group), comparison.iloc[0]["Months"])

        for _, group in monthly.groupby("policy"):
            average_weight = group["executed_stock_weight"].mean()
            expected = (
                average_weight * group["market_return"]
                + (1 - average_weight) * group["cash_return"]
            )
            pd.testing.assert_series_equal(
                group["same_exposure_return"], expected, check_names=False
            )


if __name__ == "__main__":
    unittest.main()
