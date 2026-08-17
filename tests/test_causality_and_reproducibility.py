import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiment import create_run_directory
from src.features import factor_factory
from src.modeling import (
    build_model_target,
    earliest_walk_forward_prediction_date,
    fit_latest_ews,
    make_model,
    walk_forward_predict,
)
from src.regression import walk_forward_regression
from src.splits import research_view


class CausalityAndReproducibilityTests(unittest.TestCase):
    def test_forward_drawdown_target_requires_complete_future_path(self):
        index = pd.date_range("2020-01-31", periods=8, freq="ME")
        price = pd.Series([100, 98, 94, 101, 103, 102, 104, 105], index=index)
        target = build_model_target(
            price,
            mode="future_drawdown",
            horizon=3,
            drawdown_threshold=-0.05,
        )
        self.assertEqual(target.loc[index[0], "y"], 0.0)
        self.assertEqual(target.loc[index[2], "y"], 1.0)
        self.assertTrue(target["y"].iloc[-3:].isna().all())
        self.assertAlmostEqual(
            target.loc[index[0], "future_path_drawdown"], -0.06
        )

    def test_drawdown_target_truncation_does_not_use_future_prices(self):
        index = pd.date_range("2020-01-31", periods=24, freq="ME")
        price = pd.Series(100 + np.arange(24.0), index=index)
        full = build_model_target(price, mode="future_drawdown", horizon=3)
        truncated = build_model_target(
            price.iloc[:18], mode="future_drawdown", horizon=3
        )
        pd.testing.assert_frame_equal(full.iloc[:15], truncated.iloc[:15])
        self.assertTrue(truncated["y"].iloc[-3:].isna().all())

    def test_cash_excess_target_uses_signal_date_cash_hurdle(self):
        index = pd.date_range("2020-01-31", periods=6, freq="ME")
        price = pd.Series([100.0, 100.5, 101.0, 102.0, 103.0, 104.0], index=index)
        cash_yield = pd.Series(4.0, index=index)
        target = build_model_target(
            price,
            mode="cash_excess",
            horizon=3,
            cash_yield=cash_yield,
        )
        expected_hurdle = (1.04 ** (3 / 12)) - 1.0
        self.assertAlmostEqual(target.loc[index[0], "cash_hurdle"], expected_hurdle)
        self.assertEqual(target.loc[index[0], "y"], 1.0)
        self.assertTrue(target["y"].iloc[-3:].isna().all())

    def test_cash_excess_target_requires_cash_yield(self):
        price = pd.Series(
            [100.0, 101.0, 102.0, 103.0],
            index=pd.date_range("2020-01-31", periods=4, freq="ME"),
        )
        with self.assertRaisesRegex(ValueError, "cash_yield"):
            build_model_target(price, mode="cash_excess", horizon=3)

    def test_factor_factory_truncate_recalculation(self):
        index = pd.date_range("1990-01-31", periods=140, freq="ME")
        series = pd.Series(np.exp(np.linspace(0, 1, len(index))), index=index)
        full = factor_factory(series, "x", "price")
        truncated = factor_factory(series.iloc[:100], "x", "price")
        pd.testing.assert_frame_equal(full.iloc[:100], truncated)
        self.assertFalse(any("hp" in column.lower() for column in full.columns))

    def test_holdout_mutation_cannot_change_research_view(self):
        index = pd.date_range("2010-01-31", periods=180, freq="ME")
        X = pd.DataFrame({"x": np.arange(180.0)}, index=index)
        y = pd.Series(np.arange(180.0), index=index)
        cutoff = index[119]
        left_X, left_y = research_view(X, y, cutoff)
        mutated_X, mutated_y = X.copy(), y.copy()
        mutated_X.loc[index[120]:, "x"] = -99999
        mutated_y.loc[index[120]:] = 99999
        right_X, right_y = research_view(mutated_X, mutated_y, cutoff)
        pd.testing.assert_frame_equal(left_X, right_X)
        pd.testing.assert_series_equal(left_y, right_y)

    def test_seeded_walk_forward_is_deterministic(self):
        index = pd.date_range("1990-01-31", periods=130, freq="ME")
        X = pd.DataFrame(
            {"x": np.sin(np.arange(130) / 5), "z": np.cos(np.arange(130) / 9)},
            index=index,
        )
        y = pd.Series((X["x"].shift(-3) > 0).astype(float), index=index)
        kwargs = dict(
            eval_start=index[80], eval_end=index[-4], min_train=60, purge=3,
            refit_every=3, model_type="logistic",
        )
        first = walk_forward_predict(X, y, **kwargs)
        second = walk_forward_predict(X, y, **kwargs)
        pd.testing.assert_series_equal(first, second)

    def test_mlp_is_small_seeded_and_walk_forward_deterministic(self):
        model = make_model(
            "mlp",
            mlp_params={
                "hidden_layer_sizes": (4, 2),
            },
            random_state=17,
        )
        self.assertEqual(model.named_steps["mlp"].hidden_layer_sizes, (4, 2))
        self.assertEqual(model.named_steps["mlp"].random_state, 17)

        index = pd.date_range("1990-01-31", periods=110, freq="ME")
        X = pd.DataFrame(
            {"x": np.sin(np.arange(110) / 6), "z": np.cos(np.arange(110) / 8)},
            index=index,
        )
        y = pd.Series((X["x"].shift(-3) > 0).astype(float), index=index)
        kwargs = dict(
            eval_start=index[100],
            eval_end=index[-4],
            min_train=60,
            purge=3,
            refit_every=3,
            model_type="mlp",
            mlp_params={"hidden_layer_sizes": (4, 2)},
            random_state=17,
        )
        first = walk_forward_predict(X, y, **kwargs)
        second = walk_forward_predict(X, y, **kwargs)
        pd.testing.assert_series_equal(first, second)

    def test_mlp_risk_veto_is_one_sided_and_deterministic(self):
        X = pd.DataFrame(
            {
                "x": np.linspace(-2.0, 2.0, 40),
                "z": np.sin(np.arange(40) / 3),
            }
        )
        y = pd.Series((X["x"] + 0.2 * X["z"] > 0).astype(float))
        params = {
            "hidden_layer_sizes": (2,),
            "activation": "tanh",
            "solver": "lbfgs",
            "alpha": 0.1,
            "max_iter": 500,
            "hybrid_mode": "risk_veto",
            # These diagnostic thresholds force every row through the veto.
            "risk_on_threshold": 0.0,
            "mlp_veto_threshold": 1.01,
            "neutral_probability": 0.5,
        }
        first = make_model("mlp", mlp_params=params, random_state=17).fit(X, y)
        second = make_model("mlp", mlp_params=params, random_state=17).fit(X, y)
        first_probability = first.predict_proba(X)[:, 1]
        second_probability = second.predict_proba(X)[:, 1]
        np.testing.assert_allclose(first_probability, 0.5)
        np.testing.assert_allclose(first_probability, second_probability)
        self.assertTrue(hasattr(first.named_steps["mlp"], "linear_"))
        self.assertTrue(hasattr(first.named_steps["mlp"], "mlp_"))

    def test_mlp_class_balancing_is_train_only_and_deterministic(self):
        index = pd.date_range("1990-01-31", periods=120, freq="ME")
        X = pd.DataFrame(
            {
                "x": np.sin(np.arange(120) / 7),
                "z": np.cos(np.arange(120) / 11),
            },
            index=index,
        )
        y = pd.Series((np.arange(120) % 5 == 0).astype(float), index=index)
        kwargs = dict(
            eval_start=index[90],
            eval_end=index[110],
            min_train=60,
            purge=3,
            refit_every=1,
            model_type="mlp",
            mlp_params={
                "hidden_layer_sizes": (2,),
                "solver": "lbfgs",
                "alpha": 1.0,
                "max_iter": 300,
                "balance_classes": True,
            },
            random_state=17,
        )
        first = walk_forward_predict(X, y, **kwargs)
        second = walk_forward_predict(X, y, **kwargs)
        self.assertGreater(first.notna().sum(), 0)
        pd.testing.assert_series_equal(first, second)

    def test_rolling_training_window_remains_purged(self):
        index = pd.date_range("1990-01-31", periods=120, freq="ME")
        X = pd.DataFrame(
            {"x": np.sin(np.arange(120) / 8), "z": np.cos(np.arange(120) / 9)},
            index=index,
        )
        y = pd.Series((X["x"] > 0).astype(float), index=index)
        latest = fit_latest_ews(
            X=X,
            y=y,
            features=["x", "z"],
            horizon=3,
            asof_date=index[-1],
            min_train=24,
            max_train=36,
            model_type="logistic",
        )
        self.assertEqual(latest["train_n"], 36)
        self.assertLessEqual(
            pd.Timestamp(latest["train_end"]),
            index[-1] - pd.offsets.MonthEnd(3),
        )
        with self.assertRaisesRegex(ValueError, "max_train"):
            walk_forward_predict(
                X,
                y,
                eval_start=index[100],
                min_train=60,
                max_train=36,
            )

    def test_model_inception_uses_only_availability_and_purged_history(self):
        index = pd.date_range("2000-01-31", periods=24, freq="ME")
        X = pd.DataFrame(
            {
                "early": np.arange(24.0),
                "late": [np.nan] * 6 + list(np.arange(18.0)),
            },
            index=index,
        )
        y = pd.Series(np.arange(24) % 2, index=index, dtype=float)

        inception = earliest_walk_forward_prediction_date(
            X, y, min_train=4, purge=2
        )
        self.assertEqual(inception, index[11])

        future_X = X.copy()
        future_y = y.copy()
        future_X.loc[index[15]:, :] = -99999.0
        future_y.loc[index[15]:] = 1.0
        mutated = earliest_walk_forward_prediction_date(
            future_X, future_y, min_train=4, purge=2
        )
        self.assertEqual(mutated, inception)

    def test_regression_preprocessing_is_train_only_and_purged(self):
        index = pd.date_range("1990-01-31", periods=130, freq="ME")
        X = pd.DataFrame({"x": np.sin(np.arange(130) / 8)}, index=index)
        target = pd.Series(np.cos(np.arange(130) / 11) / 10, index=index)
        prediction, audit = walk_forward_regression(
            X, target, eval_start=index[90], eval_end=index[110],
            model_type="elastic_net", min_train=60, purge=3, refit_every=2,
        )
        self.assertGreater(prediction.notna().sum(), 0)
        prediction_period = audit["prediction_date"].dt.to_period("M")
        train_period = audit["train_end"].dt.to_period("M")
        self.assertTrue(((prediction_period - train_period).apply(lambda value: value.n) >= 3).all())

    def test_run_directories_are_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            first = create_run_directory(temp, "fixed_run")
            (first / "artifact.txt").write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                create_run_directory(temp, "fixed_run")
            self.assertEqual((first / "artifact.txt").read_text(encoding="utf-8"), "preserve")

    def test_baseline_metrics_artifact_is_preserved(self):
        baseline = Path("runs/baseline_20260812_pre_position_sizing/results/model_comparison.csv")
        if not baseline.is_file():
            self.skipTest("immutable baseline artifact not present")
        metrics = pd.read_csv(baseline).set_index("model")
        self.assertAlmostEqual(metrics.loc["SVM", "auc"], 0.7308333333333333)
        self.assertAlmostEqual(metrics.loc["SVM", "accuracy"], 0.6575342465753424)


if __name__ == "__main__":
    unittest.main()
