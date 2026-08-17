import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from run_pipeline import _last_completed_month, extend_prediction_tail
from src.marcap_kospi200 import (
    build_marcap_kospi200_proxy,
    load_marcap_kospi200_proxy,
)
from src.modeling import walk_forward_predict


class MarcapKospi200ProxyTests(unittest.TestCase):
    @staticmethod
    def _panel():
        rows = []
        dates_and_returns = [
            ("2020-01-30", 0.0),
            ("2020-01-31", 0.0),
            ("2020-02-03", 1.0),
            ("2020-02-28", 1.0),
            ("2020-03-02", -1.0),
            ("2020-03-31", -1.0),
            ("2020-04-01", 50.0),
        ]
        for date, daily_return_pct in dates_and_returns:
            for rank in range(201):
                rows.append(
                    {
                        "Code": f"{rank:05d}0",
                        "Name": f"Stock {rank}",
                        "ChangesRatio": daily_return_pct,
                        "Marcap": float(1_000_000 - rank),
                        "Market": "KOSPI",
                        "observation_date": date,
                    }
                )
            rows.append(
                {
                    "Code": "999995",
                    "Name": "Preferred",
                    "ChangesRatio": 100.0,
                    "Marcap": 9_999_999.0,
                    "Market": "KOSPI",
                    "observation_date": date,
                }
            )
        return pd.DataFrame(rows)

    def test_proxy_uses_prior_month_top_200_and_excludes_incomplete_month(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "marcap.csv"
            output = root / "proxy.csv"
            metadata_path = root / "proxy.metadata.json"
            self._panel().to_csv(source, index=False)
            frame, metadata = build_marcap_kospi200_proxy(
                source,
                output,
                metadata_path,
                chunksize=350,
                asof_date="2020-04-15",
            )

            self.assertEqual(frame.index.min(), pd.Timestamp("2020-02-29"))
            self.assertEqual(frame.index.max(), pd.Timestamp("2020-03-31"))
            self.assertAlmostEqual(frame.loc["2020-02-29", "close"], 102.01)
            self.assertEqual(frame.loc["2020-02-29", "constituents"], 200)
            self.assertAlmostEqual(frame["return_coverage"].min(), 1.0)
            self.assertFalse(metadata["is_official_kospi200"])
            self.assertFalse(metadata["official_kospi200_membership_available"])

            loaded, loaded_metadata = load_marcap_kospi200_proxy(
                source, output, metadata_path
            )
            pd.testing.assert_frame_equal(
                frame, loaded, check_dtype=False, check_freq=False
            )
            self.assertEqual(loaded_metadata["output_sha256"], metadata["output_sha256"])

            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["complete_through"], "2020-03-31")

    def test_unlabelled_prediction_tail_reaches_latest_completed_month(self):
        index = pd.date_range("2010-01-31", periods=140, freq="ME")
        X = pd.DataFrame(
            {
                "x1": np.sin(np.arange(len(index)) / 5.0),
                "x2": np.cos(np.arange(len(index)) / 7.0),
            },
            index=index,
        )
        y = pd.Series((np.arange(len(index)) % 3 == 0).astype(float), index=index)
        y.iloc[-3:] = np.nan
        historical = walk_forward_predict(
            X,
            y,
            eval_start=index[-12],
            eval_end=index[-4],
            min_train=60,
            purge=3,
            refit_every=1,
            model_type="logistic",
        )
        extended = extend_prediction_tail(
            historical,
            X,
            y,
            eval_end=index[-1],
            min_train=60,
            purge=3,
            refit_every=1,
            model_type="logistic",
            model_kwargs={"random_state": 42},
        )
        self.assertEqual(extended.index.max(), index[-1])
        self.assertTrue(extended.loc[index[-3:]].notna().all())
        self.assertEqual(
            _last_completed_month(index, asof_date=index[-1]),
            index[-2],
        )


if __name__ == "__main__":
    unittest.main()
