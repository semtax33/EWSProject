import unittest
from unittest.mock import patch

import pandas as pd

from src.hankyung_eps import (
    calculate_eps_dispersion,
    fetch_company_reports,
    normalize_eps_estimates,
)


class HankyungEPSTests(unittest.TestCase):
    @staticmethod
    def _reports():
        base = {
            "BUSINESS_CODE": "005930",
            "BUSINESS_NAME": "삼성전자",
            "STOCK_SETTLEMENT_DAY": "202612",
        }
        return [
            {
                **base,
                "REPORT_IDX": 1,
                "REPORT_DATE": "2026-06-01",
                "OFFICE_NAME": "A증권",
                "REPORT_WRITER": "김분석",
                "STOCK_PRE_EPS": "1,000",
            },
            {
                **base,
                "REPORT_IDX": 2,
                "REPORT_DATE": "2026-06-10",
                "OFFICE_NAME": "A증권",
                "REPORT_WRITER": "김분석",
                "STOCK_PRE_EPS": "1,100",
            },
            {
                **base,
                "REPORT_IDX": 3,
                "REPORT_DATE": "2026-06-05",
                "OFFICE_NAME": "B증권",
                "REPORT_WRITER": "이분석",
                "STOCK_PRE_EPS": "900",
            },
            {
                **base,
                "REPORT_IDX": 5,
                "REPORT_DATE": "2026-06-06",
                "OFFICE_NAME": "C증권",
                "REPORT_WRITER": "박분석",
                "STOCK_PRE_EPS": "0",
            },
            {
                "REPORT_IDX": 4,
                "REPORT_DATE": "2002-12-31",
                "BUSINESS_CODE": "35420",
                "BUSINESS_NAME": "NAVER",
                "OFFICE_NAME": "C증권",
                "REPORT_WRITER": "박분석",
                "STOCK_SETTLEMENT_DAY1": "2002",
                "STOCK_EPS1": "(731)",
                "STOCK_SETTLEMENT_DAY2": "2003",
                "STOCK_EPS2": "1,228",
                "STOCK_SETTLEMENT_DAY3": "",
                "STOCK_EPS3": None,
            },
        ]

    def test_normalizes_current_and_legacy_schemas(self):
        frame = normalize_eps_estimates(self._reports())

        self.assertEqual(len(frame), 6)
        self.assertEqual(set(frame["stock_code"]), {"005930", "035420"})
        legacy = frame.loc[frame["report_id"].eq("4")]
        self.assertEqual(set(legacy["fiscal_period"]), {"2002", "2003"})
        self.assertEqual(
            legacy.set_index("fiscal_period").loc["2002", "eps_estimate"],
            -731.0,
        )
        zero = frame.loc[frame["report_id"].eq("5")].iloc[0]
        self.assertTrue(zero["is_zero_eps"])

    def test_uses_latest_estimate_per_brokerage_for_sample_stddev(self):
        estimates = normalize_eps_estimates(self._reports())
        summary = calculate_eps_dispersion(estimates, min_estimates=2, ddof=1)

        self.assertEqual(len(summary), 1)
        row = summary.iloc[0]
        self.assertEqual(row["stock_code"], "005930")
        self.assertEqual(row["n_estimates"], 2)
        self.assertEqual(row["n_brokerages"], 2)
        self.assertAlmostEqual(row["eps_mean"], 1000.0)
        self.assertAlmostEqual(row["eps_stddev"], 100.0 * (2.0**0.5))
        self.assertEqual(row["as_of_date"], pd.Timestamp("2026-06-10"))

        including_zero = calculate_eps_dispersion(
            estimates, min_estimates=2, ddof=1, exclude_zero=False
        ).iloc[0]
        self.assertEqual(including_zero["n_estimates"], 3)

    @patch("src.hankyung_eps._request_json")
    def test_fetches_every_page(self, request_json):
        request_json.side_effect = [
            {"data": [{"REPORT_IDX": 2}], "last_page": 2},
            {"data": [{"REPORT_IDX": 1}], "last_page": 2},
        ]

        rows = fetch_company_reports(
            "token",
            "2026-01-01",
            "2026-01-31",
            page_size=1000,
            request_delay=0,
        )

        self.assertEqual([row["REPORT_IDX"] for row in rows], [2, 1])
        self.assertEqual(request_json.call_count, 2)
        first_url = request_json.call_args_list[0].args[0]
        second_url = request_json.call_args_list[1].args[0]
        self.assertIn("page=1", first_url)
        self.assertIn("page=2", second_url)
        self.assertIn("reportType=CO", first_url)


if __name__ == "__main__":
    unittest.main()
