"""Data provenance and statistical-independence audit helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def point_in_time_audit(series_metadata):
    """Document release-lag handling separately from unrecovered vintages."""
    columns = [
        "name", "file", "freq", "availability_lag", "group",
        "release_lag_applied", "alfred_vintage_used",
        "historical_revision_safe", "availability_method", "vintage_file",
    ]
    for column in columns:
        if column not in series_metadata:
            series_metadata = series_metadata.assign(**{column: None})
    registry = series_metadata[columns].drop_duplicates("name").copy()
    for column in [
        "release_lag_applied", "alfred_vintage_used", "historical_revision_safe"
    ]:
        registry[column] = registry[column].fillna(False).astype(bool)
    registry["selection_use"] = np.where(
        registry["historical_revision_safe"],
        "strict_point_in_time_eligible",
        "research_with_revision_caveat",
    )
    registry["remediation"] = np.where(
        registry["historical_revision_safe"],
        "none",
        "replace critical revised series with ALFRED real-time vintages before deployment",
    )
    return registry.sort_values("name").reset_index(drop=True)


def selected_point_in_time_audit(
    selected_features,
    raw_catalog,
    point_in_time,
    feature_metadata=None,
):
    """Restrict strict vintage checks to sources used by the selected model."""
    selected = pd.DataFrame(
        {
            "feature": list(selected_features),
            "base": [feature.split("__", 1)[0] for feature in selected_features],
        }
    )
    catalog = raw_catalog[["name", "source", "group"]].drop_duplicates("name")
    selected = selected.merge(
        catalog.rename(
            columns={
                "name": "base",
                "source": "catalog_source",
                "group": "catalog_group",
            }
        ),
        on="base",
        how="left",
    )
    if feature_metadata is not None:
        metadata_columns = ["feature", "source", "group", "point_in_time_rule"]
        metadata = feature_metadata.copy()
        for column in metadata_columns:
            if column not in metadata:
                metadata[column] = None
        metadata = metadata[metadata_columns].drop_duplicates("feature")
        selected = selected.merge(
            metadata.rename(
                columns={
                    "source": "feature_source",
                    "group": "feature_group",
                    "point_in_time_rule": "feature_point_in_time_rule",
                }
            ),
            on="feature",
            how="left",
        )
        selected["source"] = selected["catalog_source"].fillna(
            selected["feature_source"]
        )
        selected["group"] = selected["catalog_group"].fillna(
            selected["feature_group"]
        )
    else:
        selected["source"] = selected["catalog_source"]
        selected["group"] = selected["catalog_group"]
    selected = selected.merge(
        point_in_time.rename(columns={"name": "base"}),
        on="base",
        how="left",
        suffixes=("", "_point_in_time"),
    )
    is_fred = selected["source"].eq("FRED")
    selected["alfred_applicable"] = is_fred
    selected["release_lag_status"] = np.where(
        is_fred & selected["alfred_vintage_used"].fillna(False),
        "observed_month_end_ALFRED_vintage",
        np.where(
            is_fred & selected["release_lag_applied"].fillna(False),
            "approximate_month_lag_applied",
            np.where(
                is_fred,
                "release_lag_not_verified",
                "same_month_market_data_next_month_execution",
            ),
        ),
    )
    selected["vintage_status"] = np.where(
        is_fred & selected["alfred_vintage_used"].fillna(False),
        "alfred_real_time_vintage",
        np.where(is_fred, "current_revised_history", "not_applicable_non_fred"),
    )
    selected["strict_vintage_gate_passed"] = np.where(
        is_fred,
        selected["historical_revision_safe"].fillna(False),
        True,
    ).astype(bool)
    non_fred_release_safe = selected["source"].fillna("").str.startswith(
        "derived_"
    )
    if "feature_point_in_time_rule" in selected:
        non_fred_release_safe |= selected[
            "feature_point_in_time_rule"
        ].fillna("").str.contains("next-month execution", case=False, regex=False)
    selected["release_timing_gate_passed"] = np.where(
        is_fred,
        selected["alfred_vintage_used"].fillna(False)
        & selected["release_lag_applied"].fillna(False),
        non_fred_release_safe,
    ).astype(bool)
    selected["deployment_impact"] = np.where(
        selected["strict_vintage_gate_passed"],
        "none",
        "blocks_strict_deployment",
    )
    selected["remediation"] = np.where(
        is_fred & ~selected["strict_vintage_gate_passed"],
        "supply ALFRED real-time vintages and rerun the full nested pipeline",
        "none",
    )
    return selected


def feature_group_coverage(factor_candidates, selected_features):
    selected = set(selected_features)
    rows = []
    eligible = factor_candidates.loc[factor_candidates["eligible_robust_track"]]
    for group, group_rows in eligible.groupby("group", sort=True):
        selected_rows = group_rows.loc[group_rows["feature"].isin(selected)]
        rows.append(
            {
                "group": group,
                "candidate_features": int(group_rows["feature"].nunique()),
                "candidate_bases": int(group_rows["base"].nunique()),
                "selected_features": int(selected_rows["feature"].nunique()),
                "selected_feature_names": "|".join(selected_rows["feature"].tolist()),
                "coverage_status": "selected" if not selected_rows.empty else "missing",
                "selection_rule_note": "coverage is reported; weak groups are not forced in",
            }
        )
    return pd.DataFrame(rows)


def overlapping_target_diagnostics(signal, binary_target, future_return, horizon=3):
    """Report conservative effective N and horizon-spaced signal results."""
    data = pd.concat(
        [
            signal.rename("signal"),
            binary_target.rename("binary_target"),
            future_return.rename("future_return"),
        ],
        axis=1,
    ).dropna()
    summary = pd.DataFrame(
        [
            {
                "monthly_observations": len(data),
                "target_horizon_months": horizon,
                "conservative_effective_n": int(np.ceil(len(data) / horizon)),
                "overlap_warning": (
                    "monthly forward targets overlap; ordinary month count is not independent N"
                ),
            }
        ]
    )
    rows = []
    for offset in range(horizon):
        sample = data.iloc[offset::horizon]
        auc = (
            roc_auc_score(sample["binary_target"], sample["signal"])
            if len(sample) >= 10 and sample["binary_target"].nunique() == 2
            else np.nan
        )
        rank_ic = (
            spearmanr(sample["signal"], sample["future_return"]).statistic
            if len(sample) >= 3
            and sample["signal"].nunique() > 1
            and sample["future_return"].nunique() > 1
            else np.nan
        )
        rows.append(
            {
                "offset": offset,
                "observations": len(sample),
                "start": sample.index.min() if not sample.empty else None,
                "end": sample.index.max() if not sample.empty else None,
                "auc": auc,
                "rank_ic": rank_ic,
                "pearson_ic": sample["signal"].corr(sample["future_return"]),
            }
        )
    return summary, pd.DataFrame(rows)


def market_return_role_registry(
    signal_market_file,
    investable_market_file=None,
    *,
    investable_distribution_adjusted=False,
    market_name="KOSPI200",
    market_ticker="^KS200",
    investable_instrument="KOSPI200 TR/NTR or tracking ETF adjusted return",
    investable_ticker=None,
    portfolio_return_type=None,
    portfolio_notes=None,
    portfolio_deployment_eligible=None,
):
    """Keep model target identity distinct from an investable total return."""
    return pd.DataFrame(
        [
            {
                "role": "signal_target_and_benchmark_proxy",
                "instrument": f"{market_name} price index",
                "source": str(signal_market_file),
                "ticker": market_ticker,
                "return_type": "price return",
                "available": True,
                "used_in_current_run": True,
                "deployment_eligible": False,
                "notes": "validated index history; excludes distributions and fund frictions",
            },
            {
                "role": "investable_portfolio_return",
                "instrument": investable_instrument,
                "source": str(investable_market_file) if investable_market_file else None,
                "ticker": investable_ticker,
                "return_type": portfolio_return_type or (
                    "total/investable return net of product frictions"
                ),
                "available": investable_market_file is not None,
                "used_in_current_run": investable_market_file is not None,
                "deployment_eligible": bool(
                    portfolio_deployment_eligible
                    if portfolio_deployment_eligible is not None
                    else investable_market_file is not None
                    and investable_distribution_adjusted
                ),
                "notes": portfolio_notes or (
                    "distribution-adjusted investable return source"
                    if investable_market_file is not None
                    and investable_distribution_adjusted
                    else "source exists but is not identified as adjusted/total return"
                    if investable_market_file is not None
                    else "required before portfolio results can pass the operational gate"
                ),
            },
        ]
    )


def data_freshness_audit(
    series_metadata,
    market_daily,
    breadth_cache_metadata,
    *,
    market_series_name="kospi200",
):
    """Return machine-readable freshness and schema checks without hiding stale data."""
    as_of = max(
        pd.Timestamp(series_metadata["last_date"].max()),
        pd.Timestamp(market_daily.index.max()),
        pd.Timestamp(breadth_cache_metadata["source_last_date"]),
    )
    rows = []
    for row in series_metadata.drop_duplicates("name").itertuples(index=False):
        last_date = pd.Timestamp(row.last_date)
        age_months = (as_of.to_period("M") - last_date.to_period("M")).n
        rows.append(
            {
                "dataset": row.name,
                "source_type": "FRED",
                "last_observation": last_date,
                "as_of": as_of,
                "age_months": age_months,
                "schema_passed": True,
                "freshness_passed": age_months <= max(2, int(row.availability_lag) + 1),
            }
        )
    ohlc = market_daily[["open", "high", "low", "close"]].dropna()
    market_schema = bool(
        not ohlc.empty
        and (ohlc > 0).all(axis=None)
        and (ohlc["high"] >= ohlc[["open", "close", "low"]].max(axis=1)).all()
        and (ohlc["low"] <= ohlc[["open", "close", "high"]].min(axis=1)).all()
        and market_daily["volume"].notna().any()
    )
    market_age = (as_of.to_period("M") - market_daily.index.max().to_period("M")).n
    rows.append(
        {
            "dataset": f"{market_series_name}_price_index",
            "source_type": "market_index",
            "last_observation": market_daily.index.max(),
            "as_of": as_of,
            "age_months": market_age,
            "schema_passed": market_schema,
            "freshness_passed": market_age <= 1,
        }
    )
    breadth_last = pd.Timestamp(breadth_cache_metadata["source_last_date"])
    breadth_age = (as_of.to_period("M") - breadth_last.to_period("M")).n
    rows.append(
        {
            "dataset": "korea_stock_panel",
            "source_type": "constituent_panel",
            "last_observation": breadth_last,
            "as_of": as_of,
            "age_months": breadth_age,
            "schema_passed": bool(breadth_cache_metadata.get("output_columns")),
            "freshness_passed": breadth_age <= 1,
        }
    )
    return pd.DataFrame(rows)
