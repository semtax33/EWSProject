import unittest

import pandas as pd

from src.economic_review import (
    REVIEW_COLUMNS,
    add_economic_review_drafts,
    merge_completed_review,
)


class EconomicReviewTests(unittest.TestCase):
    def _completed(self):
        return pd.DataFrame(
            [
                {
                    "feature": "term__level",
                    "economic_channel": "yield-curve growth expectations",
                    "expected_direction": "positive association expected",
                    "publication_lag_reviewed": True,
                    "duplicate_information_reviewed": True,
                    "review_status": "approved",
                    "reviewer": "human-reviewer",
                    "reviewed_at": "2026-08-14T12:00:00+09:00",
                    "notes": "reviewed against source metadata",
                }
            ]
        )

    def test_completed_review_is_upserted(self):
        registry = pd.DataFrame(columns=REVIEW_COLUMNS)
        result = merge_completed_review(registry, self._completed())
        self.assertEqual(result.loc[0, "review_status"], "approved")
        self.assertTrue(result.loc[0, "publication_lag_reviewed"])

    def test_pending_review_cannot_be_approved_by_tool(self):
        review = self._completed()
        review.loc[0, "economic_channel"] = "pending human review"
        with self.assertRaisesRegex(ValueError, "economic_channel"):
            merge_completed_review(pd.DataFrame(columns=REVIEW_COLUMNS), review)

    def test_false_lag_review_is_rejected(self):
        review = self._completed()
        review.loc[0, "publication_lag_reviewed"] = False
        with self.assertRaisesRegex(ValueError, "publication_lag_reviewed"):
            merge_completed_review(pd.DataFrame(columns=REVIEW_COLUMNS), review)

    def test_draft_suggestions_never_approve_a_review(self):
        pending = pd.DataFrame(
            [{"feature": "sp500_realized_volatility_1m__level", "review_status": "pending"}]
        )
        result = add_economic_review_drafts(pending)
        self.assertEqual(result.loc[0, "review_status"], "pending")
        self.assertEqual(
            result.loc[0, "suggestion_status"], "draft_only_not_an_approval"
        )
        self.assertIn("risk-off", result.loc[0, "suggested_expected_direction"])


if __name__ == "__main__":
    unittest.main()
