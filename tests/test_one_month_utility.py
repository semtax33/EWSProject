import unittest

import numpy as np
import pandas as pd

from src.one_month_utility import (
    build_one_month_excess_target,
    candidate_grid,
    causal_excess_volatility,
    certainty_equivalent,
    make_one_month_outer_folds,
    portfolio_path_from_signal_weights,
    portfolio_summary,
    prediction_to_weight,
    select_locked_candidate,
    walk_forward_huber,
)


class OneMonthUtilityTests(unittest.TestCase):
    def test_target_aligns_next_month_return_with_signal_date_cash(self):
        index = pd.date_range("2020-01-31", periods=4, freq="ME")
        price = pd.Series([100.0, 102.0, 101.0, 104.0], index=index)
        cash_yield = pd.Series([12.0, 6.0, 3.0, 1.0], index=index)

        target = build_one_month_excess_target(price, cash_yield)

        self.assertAlmostEqual(
            target.loc[index[0], "future_excess_return"], 0.02 - 0.01
        )
        self.assertAlmostEqual(
            target.loc[index[1], "future_excess_return"],
            101.0 / 102.0 - 1.0 - 0.005,
        )
        self.assertTrue(target.iloc[-1].isna().all())

    def test_target_truncation_cannot_change_earlier_labels(self):
        index = pd.date_range("2020-01-31", periods=20, freq="ME")
        price = pd.Series(100 + np.arange(20.0), index=index)
        cash_yield = pd.Series(4.0, index=index)
        full = build_one_month_excess_target(price, cash_yield)
        truncated = build_one_month_excess_target(
            price.iloc[:14], cash_yield.iloc[:14]
        )
        pd.testing.assert_frame_equal(full.iloc[:13], truncated.iloc[:13])
        self.assertTrue(truncated.iloc[-1].isna().all())

    def test_walk_forward_huber_is_purged_and_deterministic(self):
        index = pd.date_range("2000-01-31", periods=140, freq="ME")
        X = pd.DataFrame(
            {
                "x": np.sin(np.arange(140) / 7),
                "z": np.cos(np.arange(140) / 11),
            },
            index=index,
        )
        target = pd.Series(
            0.01 * X["x"] - 0.005 * X["z"], index=index
        )
        kwargs = dict(
            eval_start=index[100],
            eval_end=index[120],
            alpha=0.01,
            min_train=84,
            purge=1,
            refit_every=2,
        )
        first, first_audit = walk_forward_huber(X, target, **kwargs)
        second, second_audit = walk_forward_huber(X, target, **kwargs)

        pd.testing.assert_series_equal(first, second)
        pd.testing.assert_frame_equal(first_audit, second_audit)
        self.assertFalse(first.isna().any())
        self.assertTrue(
            (
                first_audit["latest_label_realization_date"]
                <= first_audit["prediction_date"]
            ).all()
        )

    def test_volatility_and_weights_are_causal_and_bounded(self):
        index = pd.date_range("2000-01-31", periods=30, freq="ME")
        price = pd.Series(
            100 * np.cumprod(1 + np.sin(np.arange(30) / 3) / 100),
            index=index,
        )
        cash_yield = pd.Series(3.0, index=index)
        full = causal_excess_volatility(price, cash_yield, window=6)
        changed_price = price.copy()
        changed_price.iloc[20:] *= np.linspace(1, 2, 10)
        changed = causal_excess_volatility(changed_price, cash_yield, window=6)
        pd.testing.assert_series_equal(full.iloc[:20], changed.iloc[:20])

        prediction = pd.Series(0.02, index=index)
        weight = prediction_to_weight(
            prediction, full, sensitivity=2.0, min_weight=0.2, max_weight=0.8
        )
        self.assertTrue(weight.dropna().between(0.2, 0.8).all())

    def test_portfolio_executes_signal_one_month_later(self):
        index = pd.date_range("2020-01-31", periods=6, freq="ME")
        price = pd.Series([100, 101, 102, 103, 104, 105], index=index, dtype=float)
        cash_yield = pd.Series(0.0, index=index)
        signal_weight = pd.Series([0.6, 0.4, 0.8], index=index[:3])

        monthly = portfolio_path_from_signal_weights(
            price,
            cash_yield,
            signal_weight,
            transaction_cost_bps=10,
            initial_executed_weight=0.5,
        )

        self.assertEqual(monthly.index[0], index[1])
        self.assertEqual(monthly.loc[index[1], "signal_date"], index[0])
        self.assertAlmostEqual(
            monthly.loc[index[1], "executed_stock_weight"], 0.6
        )
        self.assertAlmostEqual(monthly.loc[index[1], "turnover"], 0.1)
        self.assertAlmostEqual(monthly.loc[index[2], "turnover"], 0.2)

    def test_short_partial_outer_fold_is_reported_without_sharpe(self):
        index = pd.date_range("2020-01-31", periods=8, freq="ME")
        price = pd.Series(np.linspace(100, 108, len(index)), index=index)
        cash_yield = pd.Series(2.0, index=index)
        signal_weight = pd.Series(0.6, index=index[:5])
        monthly = portfolio_path_from_signal_weights(
            price,
            cash_yield,
            signal_weight,
            transaction_cost_bps=25,
        )

        summary = portfolio_summary(monthly, risk_aversion=3)

        self.assertEqual(summary["months"], 5)
        self.assertTrue(np.isnan(summary["strategy_sharpe"]))
        self.assertTrue(np.isfinite(summary["certainty_equivalent"]))

    def test_certainty_equivalent_penalizes_unstable_returns(self):
        stable = pd.Series([0.01] * 12)
        unstable = pd.Series([0.11, -0.09] * 6)
        self.assertAlmostEqual(stable.mean(), unstable.mean())
        self.assertGreater(
            certainty_equivalent(stable, risk_aversion=3),
            certainty_equivalent(unstable, risk_aversion=3),
        )

    def test_locked_selection_ignores_outer_and_holdout_columns(self):
        rows = []
        for fold in (1, 2, 3, 4):
            rows.extend(
                [
                    {
                        "fold": fold,
                        "candidate_id": "stable",
                        "alpha": 0.01,
                        "sensitivity": 1.0,
                        "volatility_window": 12,
                        "certainty_equivalent": 0.04,
                        "lower_tail_mean_10pct": -0.02,
                        "annual_turnover": 0.5,
                        "outer_sharpe": -99.0,
                        "holdout_sharpe": -99.0,
                    },
                    {
                        "fold": fold,
                        "candidate_id": "unstable",
                        "alpha": 0.10,
                        "sensitivity": 2.0,
                        "volatility_window": 6,
                        "certainty_equivalent": 0.03 + 0.04 * (fold % 2),
                        "lower_tail_mean_10pct": -0.03,
                        "annual_turnover": 1.0,
                        "outer_sharpe": 99.0,
                        "holdout_sharpe": 99.0,
                    },
                ]
            )
        data = pd.DataFrame(rows)
        selected, _ = select_locked_candidate(data, stability_penalty=0.5)
        mutated = data.copy()
        mutated["outer_sharpe"] *= -1_000
        mutated["holdout_sharpe"] *= -1_000
        selected_mutated, _ = select_locked_candidate(
            mutated, stability_penalty=0.5
        )

        self.assertEqual(selected["candidate_id"], "stable")
        self.assertEqual(
            selected["candidate_id"], selected_mutated["candidate_id"]
        )

    def test_candidate_grid_is_bounded_and_outer_folds_are_purged(self):
        grid = candidate_grid()
        self.assertEqual(len(grid), 18)
        self.assertEqual(grid["candidate_id"].nunique(), 18)

        index = pd.date_range("2000-01-31", periods=240, freq="ME")
        folds = make_one_month_outer_folds(
            index,
            research_end=index[-1],
            min_train_months=84,
            inner_validation_months=24,
            outer_validation_months=24,
        )
        self.assertGreaterEqual(len(folds), 5)
        for fold in folds:
            self.assertGreaterEqual(
                (
                    fold.inner_validation_start.to_period("M")
                    - fold.development_end.to_period("M")
                ).n,
                2,
            )
            self.assertGreaterEqual(
                (
                    fold.outer_start.to_period("M")
                    - fold.inner_validation_end.to_period("M")
                ).n,
                2,
            )


if __name__ == "__main__":
    unittest.main()
