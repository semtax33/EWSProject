import unittest

import numpy as np
import pandas as pd

from src.backtest import annual_yield_to_monthly_return, run_backtest
from src.position_sizing import (
    expanding_percentile_weight,
    fixed_bin_weight,
    linear_weight,
    smoothed_linear_weight,
    static_weight,
)


class PositionSizingTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2000-01-31", periods=8, freq="ME")

    def test_linear_matches_original_clip(self):
        score = pd.Series([0, 20, 35, 63, 80, 100, np.nan, 50], index=self.index)
        actual = linear_weight(score)
        expected = (score / 100).clip(0.20, 0.80)
        pd.testing.assert_series_equal(
            actual, expected.rename("target_stock_weight")
        )

    def test_smoothed_linear_is_causal(self):
        score = pd.Series([20, 80, 30, 70, 40, 60, 50, 90], index=self.index)
        original = smoothed_linear_weight(score, span=3)
        changed = score.copy()
        changed.iloc[5:] = [0, 0, 0]
        recalculated = smoothed_linear_weight(changed, span=3)
        pd.testing.assert_series_equal(original.iloc[:5], recalculated.iloc[:5])

    def test_static_weight_preserves_missing_signal_dates(self):
        score = pd.Series([20, np.nan, 80], index=self.index[:3])
        actual = static_weight(score, weight=0.5)
        expected = pd.Series(
            [0.5, np.nan, 0.5],
            index=self.index[:3],
            name="target_stock_weight",
        )
        pd.testing.assert_series_equal(actual, expected)

    def test_fixed_bin_boundaries(self):
        score = pd.Series([34.99, 35, 49.99, 50, 64.99, 65, np.nan, 100], index=self.index)
        actual = fixed_bin_weight(score)
        expected = pd.Series(
            [0.2, 0.4, 0.4, 0.6, 0.6, 0.8, np.nan, 0.8],
            index=self.index,
            name="target_stock_weight",
        )
        pd.testing.assert_series_equal(actual, expected)

    def test_expanding_percentile_has_no_future_leakage(self):
        score = pd.Series([10, 20, 30, 40, 50, 60, 70, 80], index=self.index)
        original = expanding_percentile_weight(score, min_history=3)
        changed_future = score.copy()
        changed_future.iloc[6:] = [-999, 999]
        changed = expanding_percentile_weight(changed_future, min_history=3)
        pd.testing.assert_series_equal(original.iloc[:6], changed.iloc[:6])
        self.assertTrue(original.iloc[:3].isna().all())

    def test_expanding_percentile_truncate_recalculation_is_identical(self):
        index = pd.date_range("2000-01-31", periods=80, freq="ME")
        score = pd.Series(np.sin(np.arange(80) / 7) * 20 + 50, index=index)
        full = expanding_percentile_weight(score, min_history=12)
        for cutoff in (24, 40, 63):
            truncated = expanding_percentile_weight(score.iloc[:cutoff], min_history=12)
            pd.testing.assert_series_equal(full.iloc[:cutoff], truncated)

    def test_expanding_percentile_constants_ties_nan_and_bounds(self):
        index = pd.date_range("2000-01-31", periods=50, freq="ME")
        score = pd.Series(50.0, index=index)
        score.iloc[20] = np.nan
        weight = expanding_percentile_weight(score, min_history=10)
        self.assertTrue(weight.iloc[:10].isna().all())
        self.assertTrue(np.isnan(weight.iloc[20]))
        valid = weight.dropna()
        self.assertTrue(valid.between(0.20, 0.80).all())
        self.assertTrue(valid.isin([0.20, 0.35, 0.50, 0.65, 0.80]).all())

    def test_cash_return_conventions(self):
        annual = pd.Series([0.0, 4.8, 12.0])
        simple = annual_yield_to_monthly_return(annual, "simple_divide_12")
        effective = annual_yield_to_monthly_return(annual, "effective_annual_compound")
        self.assertAlmostEqual(simple.iloc[1], 0.004)
        self.assertAlmostEqual(effective.iloc[2], 1.12 ** (1 / 12) - 1)
        with self.assertRaises(ValueError):
            annual_yield_to_monthly_return(annual, "unknown")

    def test_backtest_executes_target_one_month_later(self):
        price = pd.Series(np.arange(100, 108), index=self.index, dtype=float)
        score = pd.Series(np.arange(10, 90, 10), index=self.index, dtype=float)
        backtest = run_backtest(price, score, transaction_cost_bps=0, verbose=False)
        pd.testing.assert_series_equal(
            backtest["executed_stock_weight"],
            backtest["target_stock_weight"].shift(1),
            check_names=False,
        )
        first_valid = backtest["executed_stock_weight"].first_valid_index()
        self.assertEqual(backtest.loc[first_valid, "turnover"], 0.2)

    def test_first_trade_cost_uses_cash_start(self):
        price = pd.Series(np.arange(100, 108), index=self.index, dtype=float)
        score = pd.Series(20.0, index=self.index)
        backtest = run_backtest(price, score, transaction_cost_bps=10, verbose=False)
        first_valid = backtest["executed_stock_weight"].first_valid_index()
        self.assertAlmostEqual(
            backtest.loc[first_valid, "transaction_cost"],
            0.20 * 10 / 10000,
        )


if __name__ == "__main__":
    unittest.main()
