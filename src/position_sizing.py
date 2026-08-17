"""Point-in-time position-sizing policies for the EWS signal.

The functions in this module deliberately keep three concepts separate:

* ``raw_ews``: the model score on the 0--100 scale;
* ``target_stock_weight``: the allocation decided at the signal date; and
* ``executed_stock_weight``: the target applied one month later by the backtest.

Only the first two are produced here.  Execution timing belongs to the
backtest module, which prevents an accidental same-month signal trade.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _as_float_series(raw_ews: pd.Series) -> pd.Series:
    if not isinstance(raw_ews, pd.Series):
        raise TypeError("raw_ews must be a pandas Series")
    result = pd.to_numeric(raw_ews, errors="coerce").astype(float)
    result.name = "raw_ews"
    return result


def linear_weight(
    raw_ews: pd.Series,
    min_weight: float = 0.20,
    max_weight: float = 0.80,
) -> pd.Series:
    """Map EWS percentage points directly to weight, clipped to limits."""
    score = _as_float_series(raw_ews).clip(0.0, 100.0)
    weight = (score / 100.0).clip(lower=min_weight, upper=max_weight)
    weight.name = "target_stock_weight"
    return weight


def smoothed_linear_weight(
    raw_ews: pd.Series,
    min_weight: float = 0.20,
    max_weight: float = 0.80,
    span: int = 3,
) -> pd.Series:
    """Causally smooth the score before applying the linear allocation map."""
    if span < 1:
        raise ValueError("span must be positive")
    score = _as_float_series(raw_ews)
    smoothed = score.ewm(span=span, adjust=False, min_periods=1).mean()
    return linear_weight(smoothed, min_weight=min_weight, max_weight=max_weight)


def static_weight(raw_ews: pd.Series, weight: float = 0.50) -> pd.Series:
    """Return a fixed strategic allocation while preserving missing dates."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must stay between 0 and 1")
    score = _as_float_series(raw_ews)
    result = pd.Series(float(weight), index=score.index, name="target_stock_weight")
    result[score.isna()] = np.nan
    return result


def fixed_bin_weight(
    raw_ews: pd.Series,
    thresholds: Sequence[float] = (35.0, 50.0, 65.0),
    weights: Sequence[float] = (0.20, 0.40, 0.60, 0.80),
) -> pd.Series:
    """Apply stable, pre-declared score thresholds to the EWS."""
    if len(weights) != len(thresholds) + 1:
        raise ValueError("weights must have exactly one more item than thresholds")
    if list(thresholds) != sorted(thresholds):
        raise ValueError("thresholds must be sorted in ascending order")

    score = _as_float_series(raw_ews)
    values = np.select(
        [score < threshold for threshold in thresholds],
        list(weights[:-1]),
        default=weights[-1],
    ).astype(float)
    result = pd.Series(values, index=score.index, name="target_stock_weight")
    result[score.isna()] = np.nan
    return result


def expanding_percentile_weight(
    raw_ews: pd.Series,
    breaks: Sequence[float] = (0.20, 0.40, 0.60, 0.80),
    weights: Sequence[float] = (0.20, 0.35, 0.50, 0.65, 0.80),
    min_history: int = 36,
) -> pd.Series:
    """Size by the score's percentile using *strictly prior* observations.

    The current score is excluded from its reference distribution.  Ties use
    a mid-rank percentile.  Values before ``min_history`` remain missing so a
    policy comparison can use the common observable period without hidden
    defaults.
    """
    if len(weights) != len(breaks) + 1:
        raise ValueError("weights must have exactly one more item than breaks")
    if list(breaks) != sorted(breaks) or any(b <= 0 or b >= 1 for b in breaks):
        raise ValueError("breaks must be sorted values strictly between 0 and 1")
    if min_history < 1:
        raise ValueError("min_history must be positive")

    score = _as_float_series(raw_ews)
    result = pd.Series(np.nan, index=score.index, name="target_stock_weight")

    for position, value in enumerate(score.to_numpy()):
        if np.isnan(value):
            continue
        history = score.iloc[:position].dropna().to_numpy()
        if len(history) < min_history:
            continue
        percentile = (
            np.count_nonzero(history < value)
            + 0.5 * np.count_nonzero(history == value)
        ) / len(history)
        bucket = int(np.searchsorted(np.asarray(breaks), percentile, side="right"))
        result.iloc[position] = float(weights[bucket])

    return result


def target_weight_from_ews(
    raw_ews: pd.Series,
    policy: str = "linear",
    *,
    min_weight: float = 0.20,
    max_weight: float = 0.80,
    fixed_thresholds: Sequence[float] = (35.0, 50.0, 65.0),
    fixed_weights: Sequence[float] = (0.20, 0.40, 0.60, 0.80),
    percentile_breaks: Sequence[float] = (0.20, 0.40, 0.60, 0.80),
    percentile_weights: Sequence[float] = (0.20, 0.35, 0.50, 0.65, 0.80),
    percentile_min_history: int = 36,
    smoothing_span: int = 3,
    static_stock_weight: float = 0.50,
) -> pd.Series:
    """Dispatch a named policy and return its signal-date target weight."""
    if policy == "linear":
        return linear_weight(raw_ews, min_weight=min_weight, max_weight=max_weight)
    if policy == "smoothed_linear":
        return smoothed_linear_weight(
            raw_ews,
            min_weight=min_weight,
            max_weight=max_weight,
            span=smoothing_span,
        )
    if policy == "static_50_50":
        return static_weight(raw_ews, weight=static_stock_weight)
    if policy == "fixed_bin":
        return fixed_bin_weight(
            raw_ews,
            thresholds=fixed_thresholds,
            weights=fixed_weights,
        )
    if policy == "expanding_percentile":
        return expanding_percentile_weight(
            raw_ews,
            breaks=percentile_breaks,
            weights=percentile_weights,
            min_history=percentile_min_history,
        )
    raise ValueError(f"Unknown position-sizing policy: {policy}")


def allocation_frame(
    raw_ews: pd.Series,
    policy: str = "linear",
    **policy_kwargs,
) -> pd.DataFrame:
    """Return an auditable signal/target/executed allocation table."""
    score = _as_float_series(raw_ews)
    target = target_weight_from_ews(score, policy=policy, **policy_kwargs)
    return pd.DataFrame(
        {
            "raw_ews": score,
            "allocation_policy": policy,
            "target_stock_weight": target,
            "executed_stock_weight": target.shift(1),
        }
    )
