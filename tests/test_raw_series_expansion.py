import io
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.config import (
    EXACT_INDICATOR_GAP_FILE,
    MAX_FEATURES_PER_BASE,
    MAX_FEATURES_PER_GROUP,
    MIN_DISTINCT_GROUPS,
    RAW_SERIES_CATALOG_FILE,
    RAW_TOP_FEATURES_PER_BASE,
)
from src.audit import selected_point_in_time_audit
from src.data import read_investable_price_csv, to_monthly
from src.features import build_cross_asset_features, factor_factory
from src.modeling import (
    build_candidate_funnel,
    exhaustive_combination_selection,
    round_robin_group_candidates,
    screen_single_factors,
    select_required_core_features,
)
from src.raw_catalog import EXPECTED_GROUP_COUNTS, load_raw_series_catalog
from src.raw_download import download_fred_series
from src.vintage import fetch_month_end_vintage, read_monthly_vintage_series


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class RawSeriesExpansionTests(unittest.TestCase):
    def test_latest_available_vintage_uses_prior_reference_period(self):
        captured = {}

        def fake_download(params):
            captured.update(params)
            return (
                "observation_date,RSAFS\n"
                "2019-12-01,520000\n"
                "2020-01-01,525000\n",
                "https://example.invalid",
            )

        with patch("src.vintage._download_csv", side_effect=fake_download):
            row = fetch_month_end_vintage(
                "RSAFS", "2020-02-29", agg="latest_available"
            )
        self.assertEqual(row["value"], 525000)
        self.assertEqual(row["source_last_date"], pd.Timestamp("2020-01-01"))
        self.assertLess(pd.Timestamp(captured["cosd"]), pd.Timestamp("2020-01-01"))
        self.assertEqual(captured["vintage_date"], "2020-02-29")

    def test_vintage_reader_requires_hash_and_matching_month_end(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "T10Y2Y_point_in_time.csv"
            frame = pd.DataFrame(
                {
                    "series_id": ["T10Y2Y", "T10Y2Y"],
                    "observation_date": ["2020-01-31", "2020-02-29"],
                    "value": [0.1, 0.2],
                    "vintage_date": ["2020-01-31", "2020-02-29"],
                    "source_first_date": ["2020-01-02", "2020-02-03"],
                    "source_last_date": ["2020-01-31", "2020-02-28"],
                }
            )
            frame.to_csv(path, index=False)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix(".metadata.json").write_text(
                json.dumps(
                    {
                        "series_id": "T10Y2Y",
                        "sha256": digest,
                        "point_in_time_safe": True,
                    }
                ),
                encoding="utf-8",
            )
            values = read_monthly_vintage_series(
                path, expected_series_id="T10Y2Y"
            )
            self.assertEqual(values.loc["2020-02-29"], 0.2)
            path.write_text(path.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                read_monthly_vintage_series(path, expected_series_id="T10Y2Y")

    def test_required_core_selector_keeps_all_families_and_turnover_trend(self):
        scores = pd.DataFrame(
            [
                {"feature": "turnover__level", "n": 40, "auc": 0.8,
                 "auc_first": 0.8, "auc_second": 0.8, "rank_score": 0.8},
                {"feature": "turnover__slope_12m", "n": 40, "auc": 0.6,
                 "auc_first": 0.6, "auc_second": 0.6, "rank_score": 0.6},
                {"feature": "term__chg_3m", "n": 40, "auc": 0.7,
                 "auc_first": 0.7, "auc_second": 0.7, "rank_score": 0.7},
                {"feature": "corr__z_12m", "n": 40, "auc": 0.65,
                 "auc_first": 0.65, "auc_second": 0.65, "rank_score": 0.65},
                {"feature": "skew__ma_12m", "n": 40, "auc": 0.62,
                 "auc_first": 0.62, "auc_second": 0.62, "rank_score": 0.62},
            ]
        )
        selected, audit = select_required_core_features(
            scores,
            {
                "turnover_trend": ("turnover",),
                "term_spread": ("term",),
                "pairwise_correlation": ("corr",),
                "return_skew_1m": ("skew",),
            },
            min_oos_predictions=36,
            required_transform_tokens={"turnover_trend": ("slope_", "chg_")},
        )
        self.assertEqual(len(selected), 4)
        self.assertIn("turnover__slope_12m", selected)
        self.assertNotIn("turnover__level", selected)
        self.assertEqual(audit["required_family"].nunique(), 4)
    def test_catalog_has_exact_seven_basket_70_sensor_design(self):
        catalog = load_raw_series_catalog(RAW_SERIES_CATALOG_FILE)
        enabled = catalog.loc[catalog["enabled"]]
        self.assertEqual(len(enabled), 70)
        self.assertEqual(enabled.groupby("group").size().to_dict(), EXPECTED_GROUP_COUNTS)
        self.assertEqual(enabled["name"].nunique(), 70)
        for required in (
            "cpi",
            "m2",
            "high_yield_spread",
            "kospi200_return_1m",
            "korea_stock_universe_advance_decline_ratio",
            "korea_stock_universe_pairwise_correlation_1m",
            "broad_us_dollar_index",
            "aud_chf",
            "expected_inflation_10y",
        ):
            self.assertIn(required, set(enabled["name"]))

    def test_missing_exact_vendor_series_cannot_enter_selection(self):
        gaps = pd.read_csv(EXACT_INDICATOR_GAP_FILE)
        self.assertTrue(
            gaps["selection_use"].eq("prohibited_until_exact_source").all()
        )
        self.assertIn("kospi200_epr", set(gaps["indicator"]))
        self.assertIn("korea_eps_revision_up_ratio", set(gaps["indicator"]))

    def test_full_factory_design_exceeds_ten_thousand_candidates(self):
        catalog = load_raw_series_catalog(RAW_SERIES_CATALOG_FILE)
        sample = pd.Series(
            np.arange(100.0), index=pd.date_range("2000-01-31", periods=100, freq="ME")
        )
        factors_per_raw = factor_factory(sample, "sample", "index").shape[1]
        planned = int(catalog.loc[catalog["enabled"], "factory_mode"].eq("full").sum())
        self.assertEqual(planned, 70)
        self.assertGreaterEqual(planned * factors_per_raw, 10_000)

    def test_quarterly_release_lag_and_forward_fill_are_causal(self):
        source = pd.Series(
            [10.0, 20.0],
            index=pd.to_datetime(["2020-01-01", "2020-04-01"]),
        )
        monthly = to_monthly(source, "quarterly", availability_lag=4)
        self.assertEqual(monthly.index.min(), pd.Timestamp("2020-05-31"))
        self.assertEqual(monthly.loc["2020-07-31"], 10.0)
        self.assertEqual(monthly.loc["2020-08-31"], 20.0)
        self.assertNotIn(pd.Timestamp("2020-04-30"), monthly.index)

    def test_aud_chf_cross_uses_the_two_official_quote_directions(self):
        index = pd.date_range("2020-01-31", periods=2, freq="ME")
        panel = pd.DataFrame(
            {"usd_per_aud": [0.70, 0.75], "chf_per_usd": [0.90, 0.80]},
            index=index,
        )
        factors, metadata = build_cross_asset_features(panel)
        np.testing.assert_allclose(factors["aud_chf"], [0.63, 0.60])
        self.assertEqual(metadata.iloc[0]["exactness"], "exact_cross_rate")

    def test_selected_fred_source_blocks_strict_vintage_gate(self):
        catalog = pd.DataFrame(
            {"name": ["macro", "derived"], "source": ["FRED", "derived_market"], "group": ["cycle", "risk"]}
        )
        point_in_time = pd.DataFrame(
            {
                "name": ["macro"],
                "release_lag_applied": [True],
                "alfred_vintage_used": [False],
                "historical_revision_safe": [False],
            }
        )
        audit = selected_point_in_time_audit(
            ["macro__level", "derived__level"], catalog, point_in_time
        ).set_index("base")
        self.assertFalse(audit.loc["macro", "strict_vintage_gate_passed"])
        self.assertEqual(audit.loc["macro", "vintage_status"], "current_revised_history")
        self.assertTrue(audit.loc["derived", "strict_vintage_gate_passed"])
        self.assertFalse(audit.loc["macro", "release_timing_gate_passed"])
        self.assertTrue(audit.loc["derived", "release_timing_gate_passed"])

    def test_curated_market_feature_uses_its_next_month_metadata(self):
        catalog = pd.DataFrame(columns=["name", "source", "group"])
        point_in_time = pd.DataFrame(
            columns=[
                "name",
                "release_lag_applied",
                "alfred_vintage_used",
                "historical_revision_safe",
            ]
        )
        metadata = pd.DataFrame(
            {
                "feature": ["nasdaq100_momentum_6m"],
                "source": ["Yahoo Finance ^NDX daily OHLCV"],
                "group": ["market_internal"],
                "point_in_time_rule": ["month-end data, next-month execution"],
            }
        )
        audit = selected_point_in_time_audit(
            ["nasdaq100_momentum_6m"],
            catalog,
            point_in_time,
            feature_metadata=metadata,
        )
        self.assertTrue(audit.loc[0, "strict_vintage_gate_passed"])
        self.assertTrue(audit.loc[0, "release_timing_gate_passed"])

    def test_investable_reader_requires_adjusted_semantics_for_strict_gate(self):
        index = pd.date_range("2020-01-01", periods=3, freq="D")
        with tempfile.TemporaryDirectory() as temp_dir:
            adjusted_path = Path(temp_dir) / "adjusted.csv"
            plain_path = Path(temp_dir) / "plain.csv"
            pd.DataFrame(
                {"Date": index, "adjusted_close": [100, 101, 102]}
            ).to_csv(adjusted_path, index=False)
            pd.DataFrame({"Date": index, "close": [100, 101, 102]}).to_csv(
                plain_path, index=False
            )
            adjusted = read_investable_price_csv(adjusted_path)
            plain = read_investable_price_csv(plain_path)
        self.assertTrue(adjusted.attrs["distribution_adjusted"])
        self.assertFalse(plain.attrs["distribution_adjusted"])

    def test_exhaustive_search_allows_multiple_transforms_from_same_raw(self):
        index = pd.date_range("2000-01-31", periods=24, freq="ME")
        target = pd.Series(np.tile([0.0, 1.0], 12), index=index)
        prediction = pd.Series(
            np.where(target.eq(1.0), 0.8, 0.2), index=index
        )
        candidates = ["a__level", "a__chg_1m"]
        features = pd.DataFrame(
            {candidate: np.arange(len(index)) for candidate in candidates},
            index=index,
        )
        self.assertIsNone(MAX_FEATURES_PER_BASE)
        with patch("src.modeling.walk_forward_predict", return_value=prediction):
            selected, _ = exhaustive_combination_selection(
                X=features,
                y=target,
                candidates=candidates,
                validation_start=index.min(),
                validation_end=index.max(),
                min_train=4,
                horizon=1,
                min_features=2,
                max_features=2,
                max_features_per_base=None,
            )
        self.assertEqual(set(selected), set(candidates))

    def test_exhaustive_search_allows_two_features_from_same_group(self):
        index = pd.date_range("2000-01-31", periods=24, freq="ME")
        target = pd.Series(np.tile([0.0, 1.0], 12), index=index)
        prediction = pd.Series(
            np.where(target.eq(1.0), 0.8, 0.2), index=index
        )
        candidates = ["a__level", "b__level"]
        features = pd.DataFrame(
            {candidate: np.arange(len(index)) for candidate in candidates},
            index=index,
        )
        self.assertIsNone(MAX_FEATURES_PER_GROUP)
        with patch("src.modeling.walk_forward_predict", return_value=prediction):
            selected, _ = exhaustive_combination_selection(
                X=features,
                y=target,
                candidates=candidates,
                validation_start=index.min(),
                validation_end=index.max(),
                min_train=4,
                horizon=1,
                min_features=2,
                max_features=2,
                feature_groups={"a": "global_risk", "b": "global_risk"},
                max_features_per_group=MAX_FEATURES_PER_GROUP,
            )
        self.assertEqual(set(selected), set(candidates))

    def test_candidate_funnel_keeps_top_three_per_raw_and_balances_groups(self):
        scores = pd.DataFrame(
            {
                "feature": [
                    "a__1", "a__2", "a__3", "a__4",
                    "b__1", "b__2", "c__1",
                ],
                "rank_score": [0.9, 0.8, 0.7, 0.6, 0.85, 0.75, 0.65],
            }
        )
        raw_stage, group_stage = build_candidate_funnel(
            scores,
            {"a": "risk", "b": "risk", "c": "cycle"},
            max_per_base=RAW_TOP_FEATURES_PER_BASE,
            max_per_group=3,
        )
        self.assertEqual(raw_stage.groupby("base").size().max(), 3)
        self.assertNotIn("a__4", set(raw_stage["feature"]))
        self.assertLessEqual(group_stage.groupby("group").size().max(), 3)

    def test_group_diversity_constraints_are_enforced(self):
        index = pd.date_range("2000-01-31", periods=24, freq="ME")
        target = pd.Series(np.tile([0.0, 1.0], 12), index=index)
        prediction = pd.Series(np.where(target.eq(1.0), 0.8, 0.2), index=index)
        candidates = ["a__x", "b__x", "c__x", "d__x"]
        groups = {"a": "risk", "b": "risk", "c": "cycle", "d": "credit"}
        features = pd.DataFrame(
            {candidate: np.arange(len(index)) for candidate in candidates},
            index=index,
        )
        self.assertEqual(MIN_DISTINCT_GROUPS, 3)
        with patch("src.modeling.walk_forward_predict", return_value=prediction):
            selected, _ = exhaustive_combination_selection(
                X=features,
                y=target,
                candidates=candidates,
                validation_start=index.min(),
                validation_end=index.max(),
                min_train=4,
                horizon=1,
                min_features=4,
                max_features=4,
                feature_groups=groups,
                max_features_per_group=MAX_FEATURES_PER_GROUP,
                min_distinct_groups=MIN_DISTINCT_GROUPS,
            )
        self.assertEqual(set(selected), set(candidates))
        balanced = round_robin_group_candidates(
            ["a__x", "b__x", "c__x", "d__x"], groups, max_features=3
        )
        self.assertEqual(balanced, ["a__x", "c__x", "d__x"])

    def test_fast_screen_is_causal_and_scores_every_feature(self):
        rng = np.random.default_rng(7)
        index = pd.date_range("2000-01-31", periods=150, freq="ME")
        signal = rng.normal(size=len(index))
        target = pd.Series(
            (signal + rng.normal(scale=0.8, size=len(index)) > 0).astype(float),
            index=index,
        )
        features = pd.DataFrame(
            {
                "signal__level": signal,
                "noise__level": rng.normal(size=len(index)),
                "signal__chg_1m": pd.Series(signal, index=index).diff(),
            },
            index=index,
        )
        arguments = {
            "y": target,
            "dev_end": index[129],
            "eval_start": index[90],
            "min_train": 60,
            "horizon": 3,
            "refit_every": 3,
            "model_type": "fast_logistic",
        }
        first = screen_single_factors(X=features, **arguments).set_index("feature")
        mutated = features.copy()
        mutated.loc[index[130]:, :] = 1e9
        second = screen_single_factors(X=mutated, **arguments).set_index("feature")

        self.assertEqual(set(first.index), set(features.columns))
        pd.testing.assert_frame_equal(first.sort_index(), second.sort_index())
        self.assertGreater(
            first.loc["signal__level", "auc"],
            first.loc["noise__level", "auc"],
        )

    def test_fred_download_is_validated_and_written_atomically(self):
        payload = b"observation_date,TEST\n2020-01-01,1.0\n2020-02-01,2.0\n"

        def opener(_request, timeout):
            self.assertGreater(timeout, 0)
            return _Response(payload)

        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "TEST.csv"
            result = download_fred_series(
                "TEST", destination, opener=opener, retries=1
            )
            self.assertEqual(result["status"], "downloaded")
            self.assertEqual(result["observations"], 2)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(destination.with_suffix(".csv.part").exists())


if __name__ == "__main__":
    unittest.main()
