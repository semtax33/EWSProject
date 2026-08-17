import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.frozen_scoring import load_frozen_scoring_spec, score_frozen_mlp
from src.shadow import LEDGER_COLUMNS, canonical_spec_hash


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FrozenScoringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp.name)
        dates = pd.date_range("2016-01-31", "2020-01-31", freq="ME")
        step = np.arange(len(dates), dtype=float)
        factors = pd.DataFrame(
            {
                "a": np.sin(step / 4.0) + step / 100.0,
                "b": np.cos(step / 5.0) - step / 200.0,
            },
            index=dates,
        )
        factors.index.name = "observation_date"
        factors.to_parquet(self.run_dir / "factor_matrix.parquet")
        target = pd.DataFrame(
            {
                "future_return": np.where(step % 2, 0.03, -0.02),
                "y": (step % 2).astype(float),
            },
            index=dates,
        )
        target.index.name = "observation_date"
        target.to_csv(self.run_dir / "target.csv")
        signal = pd.Series(100 + step, index=dates, name="kospi200")
        signal.to_csv(self.run_dir / "kospi200_monthly.csv", index_label="observation_date")
        portfolio = pd.Series(
            100 + step * 1.2,
            index=dates,
            name="investable_kospi200",
        )
        portfolio.to_csv(
            self.run_dir / "portfolio_return_source_monthly.csv",
            index_label="observation_date",
        )
        panel = pd.DataFrame({"cash_yield_3m": 3.0}, index=dates)
        panel.index.name = "observation_date"
        panel.to_parquet(self.run_dir / "monthly_panel.parquet")
        history = pd.DataFrame(
            {
                "raw_ews": [40.0, 50.0, 60.0],
                "target_stock_weight": [0.4, 0.5, 0.6],
            },
            index=pd.date_range("2019-10-31", "2019-12-31", freq="ME"),
        )
        history.index.name = "observation_date"
        history_path = self.run_dir / "mlp_frozen_score_history.csv"
        history.to_csv(history_path)
        pd.DataFrame(columns=LEDGER_COLUMNS).to_csv(
            self.run_dir / "mlp_research_shadow_ledger.csv", index=False
        )
        (self.run_dir / "experiment_manifest.json").write_text(
            json.dumps({"status": "complete", "run_id": "synthetic"}),
            encoding="utf-8",
        )
        self.spec = {
            "schema_version": 1,
            "status": "research_shadow_only",
            "capital_authorized": False,
            "freeze_date": "2019-12-31",
            "first_eligible_observation": "2020-01-31",
            "features": ["a", "b"],
            "model": "mlp",
            "scoring_protocol": "monthly_expanding_refit_v1",
            "forecast_horizon_months": 3,
            "minimum_training_months": 12,
            "random_state": 42,
            "model_params": {
                "hidden_layer_sizes": [2],
                "solver": "lbfgs",
                "alpha": 1.0,
                "max_iter": 300,
            },
            "allocation_policy": "linear",
            "sizing_config": {"min_weight": 0.2, "max_weight": 0.8},
            "stock_weight_limits": [0.2, 0.8],
            "transaction_cost_bps": 10,
            "cash_return_convention": "simple_divide_12",
            "same_exposure_stock_weight": 0.5,
            "factor_matrix_file": "factor_matrix.parquet",
            "target_file": "target.csv",
            "signal_market_file": "kospi200_monthly.csv",
            "signal_market_column": "kospi200",
            "portfolio_price_file": "portfolio_return_source_monthly.csv",
            "portfolio_price_column": "investable_kospi200",
            "cash_yield_file": "monthly_panel.parquet",
            "cash_yield_column": "cash_yield_3m",
            "score_history_file": "mlp_frozen_score_history.csv",
            "score_history_sha256": file_hash(history_path),
            "ledger_file": "mlp_research_shadow_ledger.csv",
            "monitoring": {
                "score_population_stability_index_warning": 0.25,
                "monthly_turnover_warning": 0.4,
                "active_drawdown_vs_same_exposure_stop": -0.1,
                "calibration_slope_warning_range": [0.5, 1.5],
                "required_realized_fields": [
                    "strategy_return",
                    "same_exposure_return",
                    "active_return",
                ],
            },
        }
        self._write_spec()

    def tearDown(self):
        self.temp.cleanup()

    def _write_spec(self):
        self.spec["freeze_hash"] = canonical_spec_hash(self.spec)
        (self.run_dir / "mlp_research_shadow_spec.json").write_text(
            json.dumps(self.spec), encoding="utf-8"
        )

    def test_frozen_score_uses_one_month_execution_delay_and_realized_returns(self):
        packet = score_frozen_mlp(self.run_dir, asof_date="2020-01-31")
        self.assertTrue(packet["appendable_to_shadow_ledger"])
        self.assertEqual(packet["monitor_status"], "pass")
        self.assertAlmostEqual(packet["executed_stock_weight"], 0.6)
        self.assertAlmostEqual(packet["prior_executed_stock_weight"], 0.5)
        self.assertAlmostEqual(packet["turnover"], 0.1)
        self.assertEqual(packet["label_cutoff_date"], "2019-10-31")
        self.assertLessEqual(packet["training_end"], packet["label_cutoff_date"])
        self.assertAlmostEqual(
            packet["active_return"],
            packet["strategy_return"] - packet["same_exposure_return"],
        )

    def test_future_labels_after_purge_cutoff_do_not_change_score(self):
        first = score_frozen_mlp(self.run_dir, asof_date="2020-01-31")
        target_path = self.run_dir / "target.csv"
        target = pd.read_csv(target_path, index_col=0, parse_dates=True)
        target.loc[target.index > "2019-10-31", "y"] = 1 - target.loc[
            target.index > "2019-10-31", "y"
        ]
        target.to_csv(target_path)
        second = score_frozen_mlp(self.run_dir, asof_date="2020-01-31")
        self.assertAlmostEqual(first["raw_ews"], second["raw_ews"], places=12)

    def test_spec_and_score_history_hashes_are_fail_closed(self):
        tampered_spec = dict(self.spec, allocation_policy="fixed_bin")
        path = self.run_dir / "tampered.json"
        path.write_text(json.dumps(tampered_spec), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_frozen_scoring_spec(path)

        history_path = self.run_dir / "mlp_frozen_score_history.csv"
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write("2020-02-29,70,0.7\n")
        with self.assertRaisesRegex(ValueError, "score-history artifact hash mismatch"):
            score_frozen_mlp(self.run_dir, asof_date="2020-01-31")

    def test_missing_or_duplicate_prior_shadow_month_is_rejected(self):
        extended = pd.read_csv(
            self.run_dir / "kospi200_monthly.csv", index_col=0, parse_dates=True
        )
        extended.loc[pd.Timestamp("2020-02-29")] = 150.0
        extended.to_csv(self.run_dir / "kospi200_monthly.csv")
        for filename, column, value in (
            ("portfolio_return_source_monthly.csv", "investable_kospi200", 160.0),
        ):
            frame = pd.read_csv(self.run_dir / filename, index_col=0, parse_dates=True)
            frame.loc[pd.Timestamp("2020-02-29"), column] = value
            frame.to_csv(self.run_dir / filename)
        factors = pd.read_parquet(self.run_dir / "factor_matrix.parquet")
        factors.loc[pd.Timestamp("2020-02-29")] = [0.1, 0.2]
        factors.to_parquet(self.run_dir / "factor_matrix.parquet")
        with self.assertRaisesRegex(ValueError, "Missing prior frozen shadow score"):
            score_frozen_mlp(self.run_dir, asof_date="2020-02-29")

        duplicate = pd.DataFrame(
            [
                {
                    **{column: np.nan for column in LEDGER_COLUMNS},
                    "observation_date": "2020-01-31",
                    "freeze_hash": self.spec["freeze_hash"],
                    "raw_ews": 55.0,
                    "monitor_status": "pass",
                },
                {
                    **{column: np.nan for column in LEDGER_COLUMNS},
                    "observation_date": "2020-01-31",
                    "freeze_hash": self.spec["freeze_hash"],
                    "raw_ews": 56.0,
                    "monitor_status": "pass",
                },
            ]
        )
        duplicate.to_csv(self.run_dir / "mlp_research_shadow_ledger.csv", index=False)
        with self.assertRaisesRegex(ValueError, "duplicate observation months"):
            score_frozen_mlp(self.run_dir, asof_date="2020-02-29")


if __name__ == "__main__":
    unittest.main()
