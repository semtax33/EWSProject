import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmark_visualize import (
    plot_baseline_comparison,
    plot_benchmark_allocation_dashboard,
    plot_candidate_selection,
    plot_outperformance_diagnostics,
)
from src.visualize import plot_model_comparison


class BenchmarkVisualizationTests(unittest.TestCase):
    def test_all_research_charts_render(self):
        index = pd.date_range("2017-01-31", periods=60, freq="ME")
        market_return = pd.Series(np.sin(np.arange(60) / 5) / 30, index=index)
        strategy_return = market_return * 0.7 + 0.003
        backtest = pd.DataFrame(
            {
                "strategy_return": strategy_return,
                "market_return": market_return,
                "executed_stock_weight": np.where(np.arange(60) % 5, 1.0, 0.0),
            },
            index=index,
        )
        report = {
            "policy": "absolute_momentum_3m_0_100",
            "strategy_CAGR": 0.12,
            "market_CAGR": 0.08,
            "strategy_Sharpe": 0.6,
            "market_Sharpe": 0.3,
            "strategy_MaxDrawdown": -0.25,
            "market_MaxDrawdown": -0.45,
        }
        rolling = pd.concat(
            [
                pd.DataFrame(
                    {
                        "window_end": index[window - 1 :],
                        "active_total_return": np.linspace(-0.1, 0.2, 61 - window),
                        "window_months": window,
                    }
                )
                for window in (12, 36, 60)
            ],
            ignore_index=True,
        )
        calendar = pd.DataFrame(
            {
                "year": [2017, 2018, 2019, 2020, 2021],
                "months": [12, 12, 12, 12, 12],
                "active_return": [0.02, -0.03, 0.01, -0.01, 0.04],
            }
        )
        regime = pd.DataFrame(
            {
                "regime": ["market_up_month", "market_down_month", "high_volatility", "low_volatility"],
                "annualized_active_return": [-0.1, 0.2, 0.03, 0.01],
            }
        )
        policies = [
            "absolute_momentum_3m_0_100",
            "absolute_momentum_3m_20_100",
            "absolute_momentum_12m_0_100",
            "absolute_momentum_12m_20_100",
            "sma_10m_0_100",
            "sma_12m_0_100",
        ]
        comparison = pd.DataFrame(
            {
                "policy": policies,
                "period": "pre2020_research",
                "CAGR_difference": np.linspace(0.01, 0.04, 6),
                "drawdown_improvement": np.linspace(0.05, 0.20, 6),
                "rolling_36m_market_win_ratio": np.linspace(0.4, 0.7, 6),
            }
        )
        artifact = pd.DataFrame(
            {
                "metric": ["holdout_auc", "holdout_rank_score"],
                "baseline": [0.70, 0.68],
                "current": [0.75, 0.72],
            }
        )
        same_period = pd.DataFrame(
            {
                "specification": ["baseline_20260812_fixed", "current_global_fixed"],
                "dynamic_Sharpe": [0.8, 0.9],
                "same_exposure_Sharpe": [0.75, 0.82],
                "active_Sharpe_delta": [0.05, 0.08],
                "dynamic_CAGR": [0.15, 0.17],
                "same_exposure_CAGR": [0.14, 0.15],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            plot_benchmark_allocation_dashboard(backtest, report, output / "dashboard.png")
            plot_outperformance_diagnostics(rolling, calendar, regime, output / "diagnostics.png")
            plot_candidate_selection(comparison, policies[0], output / "candidates.png")
            plot_baseline_comparison(artifact, same_period, output / "baseline.png")
            model_metrics = pd.DataFrame(
                {"model": ["Logistic", "SVM", "MLP"], "auc": [0.72, 0.58, 0.66]}
            )
            model_performance = pd.DataFrame(
                {
                    "model": ["Logistic", "SVM", "MLP"],
                    "strategy": [
                        "Logistic Dynamic", "SVM Dynamic", "MLP Dynamic"
                    ],
                    "Sharpe": [1.0, 0.7, 0.9],
                    "MaxDrawdown": [-0.12, -0.2, -0.1],
                }
            )
            plot_model_comparison(
                model_metrics,
                model_performance,
                output / "model_comparison.png",
            )
            for name in (
                "dashboard.png", "diagnostics.png", "candidates.png",
                "baseline.png", "model_comparison.png",
            ):
                self.assertGreater((output / name).stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()
