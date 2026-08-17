import unittest

import numpy as np
import pandas as pd

from src.market_breadth import _average_pairwise_correlation, market_breadth_metadata
from src.shadow import canonical_spec_hash, monitor_observation, validate_frozen_spec


def ready_spec():
    spec = {
        "status": "ready_for_next_observation",
        "freeze_date": "2026-07-31",
        "first_eligible_observation": "2026-08-31",
        "features": ["a", "b"],
        "allocation_policy": "linear",
        "monitoring": {
            "score_population_stability_index_warning": 0.25,
            "monthly_turnover_warning": 0.40,
            "active_drawdown_vs_same_exposure_stop": -0.10,
            "calibration_slope_warning_range": [0.5, 1.5],
        },
    }
    spec["freeze_hash"] = canonical_spec_hash(spec)
    return spec


class ShadowAndBreadthTests(unittest.TestCase):
    def test_freeze_hash_and_monitor(self):
        spec = ready_spec()
        self.assertTrue(validate_frozen_spec(spec))
        tampered = dict(spec, allocation_policy="fixed_bin")
        with self.assertRaises(ValueError):
            validate_frozen_spec(tampered)
        row = monitor_observation(
            spec,
            {
                "observation_date": "2026-08-31",
                "feature_values": {"a": 1, "b": 2},
                "raw_ews": 60,
                "target_stock_weight": 0.6,
                "executed_stock_weight": 0.5,
                "turnover": 0.1,
                "active_return": 0.01,
            },
        )
        self.assertEqual(row["monitor_status"], "pass")

    def test_blocked_shadow_and_missing_feature_stop(self):
        spec = ready_spec()
        spec["status"] = "blocked_until_all_deployment_gates_pass"
        spec["freeze_hash"] = canonical_spec_hash(spec)
        row = monitor_observation(
            spec,
            {
                "observation_date": "2026-08-31",
                "feature_values": {"a": 1},
                "raw_ews": 60,
                "target_stock_weight": 0.6,
            },
        )
        self.assertEqual(row["monitor_status"], "stop")
        self.assertIn("missing_frozen_feature", row["stop_reasons"])

    def test_research_only_shadow_is_recordable_but_never_authorizes_capital(self):
        spec = ready_spec()
        spec["status"] = "research_shadow_only"
        spec["capital_authorized"] = False
        spec["freeze_hash"] = canonical_spec_hash(spec)
        row = monitor_observation(
            spec,
            {
                "observation_date": "2026-08-31",
                "feature_values": {"a": 1, "b": 2},
                "raw_ews": 60,
                "target_stock_weight": 0.6,
                "turnover": 0.1,
                "active_return": 0.01,
            },
        )
        self.assertEqual(row["monitor_status"], "pass")

        unsafe = dict(spec, capital_authorized=True)
        unsafe["freeze_hash"] = canonical_spec_hash(unsafe)
        rejected = monitor_observation(
            unsafe,
            {
                "observation_date": "2026-08-31",
                "feature_values": {"a": 1, "b": 2},
                "raw_ews": 60,
                "target_stock_weight": 0.6,
            },
        )
        self.assertEqual(rejected["monitor_status"], "stop")
        self.assertIn(
            "research_shadow_cannot_authorize_capital",
            rejected["stop_reasons"],
        )

    def test_strategy_specific_zero_to_one_weight_limits(self):
        spec = ready_spec()
        spec["stock_weight_limits"] = [0.0, 1.0]
        spec["freeze_hash"] = canonical_spec_hash(spec)
        row = monitor_observation(
            spec,
            {
                "observation_date": "2026-08-31",
                "feature_values": {"a": 1, "b": 2},
                "raw_ews": 0,
                "target_stock_weight": 0.0,
                "turnover": 0.0,
                "active_return": 0.0,
            },
        )
        self.assertEqual(row["monitor_status"], "pass")

    def test_realized_return_requirements_and_active_return_consistency(self):
        spec = ready_spec()
        spec["monitoring"]["required_realized_fields"] = [
            "strategy_return",
            "same_exposure_return",
            "active_return",
        ]
        spec["freeze_hash"] = canonical_spec_hash(spec)
        base = {
            "observation_date": "2026-08-31",
            "feature_values": {"a": 1, "b": 2},
            "raw_ews": 60,
            "target_stock_weight": 0.6,
        }
        missing = monitor_observation(spec, base)
        self.assertEqual(missing["monitor_status"], "stop")
        self.assertIn("missing_realized_field", missing["stop_reasons"])
        inconsistent = monitor_observation(
            spec,
            {
                **base,
                "strategy_return": 0.02,
                "same_exposure_return": 0.01,
                "active_return": 0.02,
            },
        )
        self.assertEqual(inconsistent["monitor_status"], "stop")
        self.assertIn("active_return_inconsistent", inconsistent["stop_reasons"])

    def test_pairwise_formula_matches_explicit_correlation(self):
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        rng = np.random.default_rng(42)
        wide = pd.DataFrame(rng.normal(size=(20, 4)), index=dates, columns=list("ABCD"))
        long = wide.stack().rename("return_1d").reset_index()
        long.columns = ["observation_date", "Code", "return_1d"]
        actual, assets, days = _average_pairwise_correlation(long)
        explicit = wide.corr().to_numpy()
        expected = (explicit.sum() - len(explicit)) / (len(explicit) * (len(explicit) - 1))
        self.assertAlmostEqual(actual, expected)
        self.assertEqual((assets, days), (4, 20))

    def test_raw_outlier_diagnostics_are_not_model_eligible(self):
        index = pd.date_range("2020-01-31", periods=2, freq="ME")
        factors = pd.DataFrame(
            {
                "korea_stock_universe_return_skew_1m": [0.1, 0.2],
                "korea_stock_universe_return_dispersion_1m": [0.1, 0.2],
                "korea_stock_universe_return_skew_1m_raw": [10, 20],
                "korea_stock_universe_return_dispersion_1m_raw": [10, 20],
                "korea_stock_universe_return_up_ratio_1m": [0.5, 0.6],
                "korea_stock_universe_return_observations": [800, 800],
                "korea_stock_universe_trading_volume_ratio_12m": [1, 1],
                "korea_stock_universe_advance_decline_ratio": [1, 1],
                "korea_stock_universe_mcclellan_oscillator": [0, 0],
                "korea_stock_universe_trading_value_to_market_cap": [0.1, 0.1],
                "korea_stock_universe_share_turnover_ratio": [0.1, 0.1],
                "korea_stock_universe_pairwise_correlation_1m": [0.2, 0.2],
                "korea_stock_universe_pairwise_assets": [800, 800],
            },
            index=index,
        )
        metadata = market_breadth_metadata(factors).set_index("feature")
        self.assertFalse(metadata.loc["korea_stock_universe_return_skew_1m_raw", "model_eligible"])
        self.assertTrue(metadata.loc["korea_stock_universe_return_skew_1m", "model_eligible"])


if __name__ == "__main__":
    unittest.main()
