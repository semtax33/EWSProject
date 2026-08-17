"""Validated human-review registry updates for deployable EWS features."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


REVIEW_COLUMNS = [
    "feature",
    "economic_channel",
    "expected_direction",
    "publication_lag_reviewed",
    "duplicate_information_reviewed",
    "review_status",
    "reviewer",
    "reviewed_at",
    "notes",
]


_DRAFT_REVIEW_BY_BASE = {
    "us_corporate_equity_value": (
        "equity valuation and aggregate risk-appetite channel",
        "value above its recent trend is provisionally risk-on; verify regime sensitivity",
    ),
    "term_spread_10y3m": (
        "yield-curve growth and monetary-policy expectations",
        "steepening is provisionally risk-on; inversion and recession regimes require review",
    ),
    "term_spread_10y2y": (
        "yield-curve growth and monetary-policy expectations",
        "a higher slope is provisionally risk-on; inversion regimes require review",
    ),
    "usd_per_aud": (
        "global growth, commodity demand and cross-asset risk appetite",
        "AUD strength versus USD is provisionally risk-on",
    ),
    "us_nonfinancial_profits_after_tax": (
        "corporate earnings-cycle strength and uncertainty",
        "higher profit volatility is provisionally risk-off",
    ),
}


def economic_review_draft(feature: str) -> dict[str, str]:
    """Return non-authoritative suggestions without approving a feature.

    These fields are an analyst aid only.  They never populate the authoritative
    review columns and therefore cannot make an operational gate pass.
    """
    feature = str(feature)
    base = feature.split("__", 1)[0]
    if base in _DRAFT_REVIEW_BY_BASE:
        channel, direction = _DRAFT_REVIEW_BY_BASE[base]
    elif base.endswith("trading_volume_ratio_12m"):
        channel = "market participation and trading-liquidity channel"
        direction = "higher participation is provisionally risk-on; verify turnover spikes"
    elif base.endswith("trading_value_to_market_cap"):
        channel = "market participation and turnover-intensity channel"
        direction = "rising turnover is provisionally risk-on; verify speculative-volume regimes"
    elif base.endswith("pairwise_correlation_1m"):
        channel = "cross-sectional crowding and market-breadth channel"
        direction = "higher correlation is provisionally risk-off because diversification narrows"
    elif base.endswith("return_skew_1m"):
        channel = "cross-sectional return-asymmetry and breadth channel"
        direction = "direction is transformation-dependent and requires coefficient review"
    elif base.endswith("realized_volatility_1m"):
        channel = "realized market-risk and deleveraging channel"
        direction = "higher realized volatility is provisionally risk-off"
    elif base.endswith("downside_volatility_1m"):
        channel = "downside-tail risk and forced-deleveraging channel"
        direction = "higher downside volatility is provisionally risk-off"
    elif base.endswith("momentum_6m"):
        channel = "medium-term price-persistence channel"
        direction = "positive momentum is provisionally risk-on"
    elif base.endswith("trend_10m"):
        channel = "price relative-to-trend regime channel"
        direction = "price above trend is provisionally risk-on"
    else:
        channel = "no automatic economic-channel suggestion available"
        direction = "human reviewer must determine sign and regime dependence"
    return {
        "suggested_economic_channel": channel,
        "suggested_expected_direction": direction,
        "suggestion_status": "draft_only_not_an_approval",
    }


def add_economic_review_drafts(review: pd.DataFrame) -> pd.DataFrame:
    """Attach clearly labelled draft aids while preserving human-only fields."""
    result = review.copy()
    drafts = result["feature"].map(economic_review_draft).apply(pd.Series)
    for column in drafts.columns:
        result[column] = drafts[column].to_numpy()
    return result


def _reviewed_true(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes", "y"}
        if pd.notna(value)
        else False
    )


def validate_completed_review(review: pd.DataFrame) -> pd.DataFrame:
    """Reject incomplete or synthetic-looking approvals before registry write."""
    missing = set(REVIEW_COLUMNS).difference(review.columns)
    if missing:
        raise ValueError(f"Review columns missing: {sorted(missing)}")
    result = review[REVIEW_COLUMNS].copy()
    if result.empty:
        raise ValueError("Completed review contains no features")
    if result["feature"].isna().any() or result["feature"].duplicated().any():
        raise ValueError("Review features must be non-empty and unique")
    for column in ("economic_channel", "expected_direction", "reviewer"):
        text = result[column].fillna("").astype(str).str.strip()
        invalid = text.eq("") | text.str.contains("pending", case=False, regex=False)
        if invalid.any():
            raise ValueError(f"{column} must be completed for every feature")
        result[column] = text
    for column in ("publication_lag_reviewed", "duplicate_information_reviewed"):
        result[column] = _reviewed_true(result[column])
        if not result[column].all():
            raise ValueError(f"{column} must be true for every approved feature")
    result["review_status"] = (
        result["review_status"].fillna("").astype(str).str.strip().str.lower()
    )
    if not result["review_status"].eq("approved").all():
        raise ValueError("review_status must be approved for every feature")
    reviewed_at = pd.to_datetime(result["reviewed_at"], errors="coerce", utc=True)
    if reviewed_at.isna().any():
        raise ValueError("reviewed_at must be a valid timestamp for every feature")
    result["reviewed_at"] = reviewed_at.map(lambda value: value.isoformat())
    result["notes"] = result["notes"].fillna("").astype(str)
    return result


def merge_completed_review(registry: pd.DataFrame, review: pd.DataFrame) -> pd.DataFrame:
    """Upsert a validated review while preserving unrelated registry entries."""
    missing = set(REVIEW_COLUMNS).difference(registry.columns)
    if missing:
        raise ValueError(f"Registry columns missing: {sorted(missing)}")
    approved = validate_completed_review(review).set_index("feature")
    current = registry[REVIEW_COLUMNS].copy().set_index("feature")
    current = current.loc[~current.index.isin(approved.index)]
    return (
        pd.concat([current, approved])
        .reset_index()
        .sort_values("feature")
        .reset_index(drop=True)
    )


def write_registry_atomic(registry: pd.DataFrame, path: str | Path) -> None:
    path = Path(path).resolve()
    temporary = path.with_name(f".{path.name}.tmp")
    registry.to_csv(temporary, index=False)
    os.replace(temporary, path)
