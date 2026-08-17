import unittest

import numpy as np
import pandas as pd

from src.benchmark_allocation import (
    build_candidate_backtests,
    build_candidate_weights,
    evaluate_candidate_periods,
    pre2020_fold_results,
    select_benchmark_policy,
)
from src.backtest import run_backtest


class BenchmarkAllocationTests(unittest.TestCase):
    def test_candidate_weights_are_causal_under_truncation(self):
        index = pd.date_range("2000-01-31", periods=80, freq="ME")
        market = pd.Series(100 * np.exp(np.linspace(0, 1, 80)), index=index)
        full = build_candidate_weights(market)
        truncated = build_candidate_weights(market.iloc[:50])
        for name in full:
            pd.testing.assert_series_equal(full[name].iloc[:50], truncated[name])

    def test_momentum_signal_executes_one_month_later(self):
        index = pd.date_range("2000-01-31", periods=30, freq="ME")
        market = pd.Series(np.r_[np.arange(100, 115), np.arange(115, 100, -1)], index=index)
        weights = build_candidate_weights(market)["absolute_momentum_3m_0_100"]
        backtest = build_candidate_backtests(market, None)["absolute_momentum_3m_0_100"]
        pd.testing.assert_series_equal(
            backtest["executed_stock_weight"], weights.shift(1), check_names=False
        )

    def test_selection_ignores_holdout_and_full_rows(self):
        comparison = pd.DataFrame(
            [
                {"policy": "a", "period": "pre2020_research", "CAGR_difference": .02,
                 "relative_terminal_wealth": .5, "drawdown_improvement": .2},
                {"policy": "b", "period": "pre2020_research", "CAGR_difference": .01,
                 "relative_terminal_wealth": .2, "drawdown_improvement": .2},
                {"policy": "a", "period": "historical_holdout", "CAGR_difference": -.9,
                 "relative_terminal_wealth": -.9, "drawdown_improvement": -.9},
                {"policy": "b", "period": "historical_holdout", "CAGR_difference": 9,
                 "relative_terminal_wealth": 9, "drawdown_improvement": 9},
            ]
        )
        folds = pd.DataFrame(
            {
                "policy": ["a", "a", "b", "b"],
                "annualized_active_return": [.01, .02, .01, .02],
            }
        )
        selected, _ = select_benchmark_policy(comparison, folds)
        self.assertEqual(selected, "a")
        mutated = comparison.copy()
        mutated.loc[mutated["period"].ne("pre2020_research"), "relative_terminal_wealth"] *= -1000
        selected_mutated, _ = select_benchmark_policy(mutated, folds)
        self.assertEqual(selected_mutated, "a")

    def test_actual_history_selected_policy_beats_kospi_full_period(self):
        market_path = "runs/ews_full_20260813_augmented/portfolio_return_source_monthly.csv"
        panel_path = "runs/ews_full_20260813_augmented/monthly_panel.parquet"
        try:
            market = pd.read_csv(market_path, index_col=0, parse_dates=True).iloc[:, 0]
            cash = pd.read_parquet(panel_path)["cash_yield_3m"]
        except FileNotFoundError:
            self.skipTest("validated full-run artifacts not present")
        backtests = build_candidate_backtests(market, cash)
        comparison, periods = evaluate_candidate_periods(backtests)
        folds = pre2020_fold_results(backtests, periods["pre2020_research"])
        selected, _ = select_benchmark_policy(comparison, folds)
        report = comparison.loc[
            comparison["policy"].eq(selected)
            & comparison["period"].eq("full_history_diagnostic")
        ].iloc[0]
        self.assertEqual(selected, "absolute_momentum_3m_0_100")
        self.assertGreater(report["CAGR_difference"], 0)
        self.assertGreater(report["relative_terminal_wealth"], 0)
        selected_weight = build_candidate_weights(market)[selected].fillna(1.0)
        full_backtest = run_backtest(
            market,
            selected_weight * 100,
            target_stock_weight=selected_weight,
            allocation_policy=selected,
            cash_yield_annual_pct=cash,
            transaction_cost_bps=10,
            verbose=False,
        )
        full = full_backtest.loc[:"2026-04-30"].dropna(
            subset=["strategy_return", "market_return", "executed_stock_weight"]
        )
        self.assertEqual(full.index.min(), pd.Timestamp("1996-04-30"))
        relative_wealth = (
            (1 + full["strategy_return"]).prod()
            / (1 + full["market_return"]).prod()
            - 1
        )
        self.assertGreater(relative_wealth, 0)


if __name__ == "__main__":
    unittest.main()
