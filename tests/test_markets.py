import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.audit import market_return_role_registry
from src.data import read_market_daily_csv
from src.features import build_market_index_features
from src.markets import (
    get_market_profile,
    marketize_raw_catalog,
    mlp_params_for_market,
    mlp_target_spec_for_market,
    primary_target_spec_for_market,
    optional_market_families,
    predeclared_mlp_feature_provenance,
    predeclared_mlp_features,
    predeclared_primary_features,
    required_core_families,
)
from src.modeling import chronological_split
from src.raw_catalog import load_raw_series_catalog


class MarketProfilesTest(unittest.TestCase):
    def test_supported_profiles_keep_signal_and_investable_roles_separate(self):
        expected = {
            "kospi200": ("^KS200", "069500.KS"),
            "sp500": ("^GSPC", "SPY"),
            "nasdaq100": ("^NDX", "QQQ"),
        }
        for key, (signal_ticker, investable_ticker) in expected.items():
            profile = get_market_profile(key)
            self.assertEqual(profile.ticker, signal_ticker)
            self.assertEqual(profile.investable_ticker, investable_ticker)
            self.assertNotEqual(profile.signal_file, profile.investable_file)

    def test_catalog_marketizes_only_index_derived_sensors(self):
        catalog = load_raw_series_catalog("raw_series_catalog.csv")
        profile = get_market_profile("sp500")
        adapted = marketize_raw_catalog(catalog, profile)
        enabled_names = set(adapted.loc[adapted["enabled"], "name"])
        self.assertIn("sp500_return_1m", enabled_names)
        self.assertIn("sp500_downside_volatility_1m", enabled_names)
        self.assertNotIn("kospi200_return_1m", enabled_names)
        self.assertIn("korea_stock_universe_return_skew_1m", enabled_names)
        self.assertEqual(len(adapted), len(catalog))

    def test_foreign_market_core_uses_own_index_risk_variables(self):
        core = required_core_families(get_market_profile("nasdaq100"))
        self.assertEqual(
            core["realized_volatility"],
            ("nasdaq100_realized_volatility_1m",),
        )
        self.assertEqual(
            core["downside_risk"],
            ("nasdaq100_downside_volatility_1m",),
        )
        self.assertNotIn("absolute_trend", core)
        optional = optional_market_families(get_market_profile("nasdaq100"))
        self.assertIn("nasdaq100_momentum_12m", optional["absolute_trend"])

    def test_market_features_are_prefixed_and_audited_for_selected_market(self):
        dates = pd.bdate_range("2020-01-01", periods=420)
        close = pd.Series(100 * np.exp(np.linspace(0, 0.25, len(dates))), index=dates)
        daily = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.linspace(1_000_000, 1_500_000, len(dates)),
            }
        )
        factors, metadata = build_market_index_features(
            daily,
            series_name="sp500",
            market_name="S&P 500",
            ticker="^GSPC",
        )
        self.assertTrue(all(name.startswith("sp500_") for name in factors.columns))
        self.assertEqual(set(metadata["market"]), {"S&P 500"})
        self.assertEqual(set(metadata["ticker"]), {"^GSPC"})
        self.assertIn("sp500_momentum_12m", factors)
        self.assertIn("sp500_trend_10m", factors)

        roles = market_return_role_registry(
            "SP500.csv",
            "SPY_adjusted.csv",
            investable_distribution_adjusted=True,
            market_name="S&P 500",
            market_ticker="^GSPC",
            investable_instrument="SPDR S&P 500 ETF Trust",
            investable_ticker="SPY",
        )
        self.assertEqual(roles.loc[0, "instrument"], "S&P 500 price index")
        self.assertEqual(roles.loc[1, "ticker"], "SPY")
        self.assertTrue(roles.loc[1, "deployment_eligible"])

    def test_market_trend_features_are_causal_under_truncation(self):
        dates = pd.bdate_range("2018-01-01", periods=900)
        close = pd.Series(100 * np.exp(np.linspace(0, 0.40, len(dates))), index=dates)
        daily = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.linspace(1_000_000, 1_800_000, len(dates)),
            }
        )
        cutoff = pd.Timestamp("2020-12-31")
        full, _ = build_market_index_features(
            daily,
            series_name="nasdaq100",
            market_name="NASDAQ-100",
            ticker="^NDX",
        )
        truncated, _ = build_market_index_features(
            daily.loc[:cutoff],
            series_name="nasdaq100",
            market_name="NASDAQ-100",
            ticker="^NDX",
        )
        pd.testing.assert_frame_equal(
            full.loc[truncated.index, [
                "nasdaq100_momentum_6m",
                "nasdaq100_momentum_12m",
                "nasdaq100_trend_10m",
            ]],
            truncated[[
                "nasdaq100_momentum_6m",
                "nasdaq100_momentum_12m",
                "nasdaq100_trend_10m",
            ]],
        )

    def test_standard_yfinance_ohlcv_csv_is_read_without_legacy_repair(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "market.csv"
            pd.DataFrame(
                {
                    "Date": ["2024-01-02", "2024-01-03"],
                    "Open": [100.0, 101.0],
                    "High": [102.0, 103.0],
                    "Low": [99.0, 100.0],
                    "Close": [101.0, 102.0],
                    "Volume": [1_000, 1_100],
                }
            ).to_csv(path, index=False)
            result = read_market_daily_csv(path)
        self.assertEqual(list(result.columns), ["close", "high", "low", "open", "volume"])
        self.assertEqual(result.loc[pd.Timestamp("2024-01-03"), "close"], 102.0)

    def test_fixed_holdout_date_is_independent_of_market_history_length(self):
        dates = pd.date_range("1988-01-31", "2026-05-31", freq="ME")
        split = chronological_split(
            pd.Series(0.0, index=dates),
            test_start="2020-04-30",
            validation_months=72,
        )
        self.assertEqual(split["test_start"], pd.Timestamp("2020-04-30"))
        self.assertEqual(split["validation_start"], pd.Timestamp("2014-04-30"))
        self.assertEqual(split["dev_end"], pd.Timestamp("2014-03-31"))

    def test_nasdaq_uses_preholdout_small_sample_mlp(self):
        default = {
            "hidden_layer_sizes": (8, 4),
            "solver": "adam",
            "alpha": 0.05,
        }
        nasdaq = mlp_params_for_market(get_market_profile("nasdaq100"), default)
        sp500 = mlp_params_for_market(get_market_profile("sp500"), default)

        self.assertEqual(nasdaq, default)
        self.assertIsNot(nasdaq, default)
        self.assertEqual(sp500, default)
        self.assertIsNot(sp500, default)

        kospi = mlp_params_for_market(get_market_profile("kospi200"), default)
        self.assertEqual(kospi["hidden_layer_sizes"], (4,))
        self.assertEqual(kospi["solver"], "lbfgs")
        self.assertEqual(kospi["alpha"], 0.10)
        self.assertEqual(kospi["hybrid_mode"], "risk_veto")
        self.assertEqual(kospi["risk_on_threshold"], 0.65)
        self.assertEqual(kospi["mlp_veto_threshold"], 0.50)

        fixed = predeclared_mlp_features(get_market_profile("nasdaq100"))
        self.assertEqual(len(fixed), 4)
        self.assertIn("us_corporate_equity_value__dist_ma_3m", fixed)
        self.assertIn("term_spread_10y3m__ma_60m_chg_2m", fixed)
        self.assertEqual(
            mlp_target_spec_for_market(get_market_profile("nasdaq100"))["mode"],
            "future_drawdown",
        )
        self.assertEqual(
            mlp_target_spec_for_market(get_market_profile("sp500"))["mode"],
            "absolute_positive",
        )
        self.assertEqual(
            mlp_target_spec_for_market(get_market_profile("kospi200"))["mode"],
            "cash_excess",
        )
        self.assertEqual(
            primary_target_spec_for_market(get_market_profile("kospi200"))["mode"],
            "cash_excess",
        )
        self.assertEqual(
            primary_target_spec_for_market(get_market_profile("sp500"))["mode"],
            "absolute_positive",
        )
        self.assertEqual(
            len(predeclared_primary_features(get_market_profile("kospi200"))),
            4,
        )
        self.assertIsNone(
            predeclared_primary_features(get_market_profile("sp500"))
        )
        sp500_fixed = predeclared_mlp_features(get_market_profile("sp500"))
        self.assertEqual(len(sp500_fixed), 4)
        self.assertIn("term_spread_10y3m__ma_60m_chg_2m", sp500_fixed)
        self.assertEqual(
            predeclared_mlp_feature_provenance(get_market_profile("sp500")),
            "locked_pre2020_development_candidate_v1",
        )
        kospi_fixed = predeclared_mlp_features(get_market_profile("kospi200"))
        self.assertEqual(len(kospi_fixed), 4)
        self.assertIn(
            "korea_stock_universe_pairwise_correlation_1m__ewma_12m_chg_24m",
            kospi_fixed,
        )
        self.assertEqual(
            predeclared_mlp_feature_provenance(get_market_profile("kospi200")),
            "locked_pre2020_required_core_candidate_v1",
        )


if __name__ == "__main__":
    unittest.main()
