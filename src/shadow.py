"""Frozen forward-shadow specification validation and append-only ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


LEDGER_COLUMNS = [
    "observation_date",
    "recorded_at",
    "freeze_hash",
    "raw_ews",
    "target_stock_weight",
    "executed_stock_weight",
    "turnover",
    "strategy_return",
    "same_exposure_return",
    "active_return",
    "score_psi",
    "calibration_slope",
    "active_drawdown",
    "warning_reasons",
    "stop_reasons",
    "monitor_status",
]


def canonical_spec_hash(spec):
    payload = {key: value for key, value in spec.items() if key != "freeze_hash"}
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_frozen_spec(spec):
    required = {
        "status",
        "freeze_date",
        "first_eligible_observation",
        "features",
        "allocation_policy",
        "monitoring",
        "freeze_hash",
    }
    missing = required.difference(spec)
    if missing:
        raise ValueError(f"Frozen shadow spec missing fields: {sorted(missing)}")
    expected = canonical_spec_hash(spec)
    if spec["freeze_hash"] != expected:
        raise ValueError("Frozen shadow specification hash mismatch")
    return True


def initialize_shadow_ledger(path):
    path = Path(path)
    if not path.exists():
        pd.DataFrame(columns=LEDGER_COLUMNS).to_csv(path, index=False)
    return path


def population_stability_index(reference, recent, bins=10):
    reference = pd.Series(reference).dropna().astype(float)
    recent = pd.Series(recent).dropna().astype(float)
    if len(reference) < bins * 2 or len(recent) < 3:
        return np.nan
    edges = np.unique(reference.quantile(np.linspace(0, 1, bins + 1)).to_numpy())
    if len(edges) < 3:
        return 0.0 if recent.nunique() <= 1 else np.inf
    edges[0], edges[-1] = -np.inf, np.inf
    reference_share = pd.cut(reference, edges, include_lowest=True).value_counts(
        normalize=True, sort=False
    )
    recent_share = pd.cut(recent, edges, include_lowest=True).value_counts(
        normalize=True, sort=False
    )
    reference_share = reference_share.clip(lower=1e-6)
    recent_share = recent_share.clip(lower=1e-6)
    return float(((recent_share - reference_share) * np.log(recent_share / reference_share)).sum())


def monitor_observation(spec, observation, *, history=None):
    validate_frozen_spec(spec)
    history = pd.DataFrame(columns=LEDGER_COLUMNS) if history is None else history.copy()
    date = pd.Timestamp(observation["observation_date"])
    stop = []
    warnings = []
    permitted_statuses = {"ready_for_next_observation", "research_shadow_only"}
    if spec["status"] not in permitted_statuses:
        stop.append("deployment_gates_not_passed")
    if (
        spec["status"] == "research_shadow_only"
        and spec.get("capital_authorized", False)
    ):
        stop.append("research_shadow_cannot_authorize_capital")
    if date < pd.Timestamp(spec["first_eligible_observation"]):
        stop.append("observation_precedes_frozen_forward_window")
    supplied_hash = observation.get("freeze_hash", spec["freeze_hash"])
    if supplied_hash != spec["freeze_hash"]:
        stop.append("freeze_hash_mismatch")

    missing_features = [
        feature for feature in spec["features"]
        if feature not in observation.get("feature_values", {})
        or pd.isna(observation.get("feature_values", {}).get(feature))
    ]
    if missing_features:
        stop.append("missing_frozen_feature:" + "|".join(missing_features))

    raw_ews = float(observation.get("raw_ews", np.nan))
    target_weight = float(observation.get("target_stock_weight", np.nan))
    turnover = float(observation.get("turnover", np.nan))
    if not 0 <= raw_ews <= 100:
        stop.append("raw_ews_outside_0_100")
    min_weight, max_weight = spec.get("stock_weight_limits", [0.20, 0.80])
    if not float(min_weight) <= target_weight <= float(max_weight):
        stop.append("target_weight_outside_limits")
    if np.isfinite(turnover) and turnover > spec["monitoring"]["monthly_turnover_warning"]:
        warnings.append("monthly_turnover_warning")

    required_realized_fields = spec["monitoring"].get(
        "required_realized_fields", []
    )
    missing_realized_fields = [
        field
        for field in required_realized_fields
        if not np.isfinite(pd.to_numeric(observation.get(field), errors="coerce"))
    ]
    if missing_realized_fields:
        stop.append(
            "missing_realized_field:" + "|".join(missing_realized_fields)
        )

    strategy_return = pd.to_numeric(
        observation.get("strategy_return"), errors="coerce"
    )
    same_exposure_return = pd.to_numeric(
        observation.get("same_exposure_return"), errors="coerce"
    )
    supplied_active_return = pd.to_numeric(
        observation.get("active_return"), errors="coerce"
    )
    if all(
        np.isfinite(value)
        for value in (
            strategy_return,
            same_exposure_return,
            supplied_active_return,
        )
    ) and not np.isclose(
        supplied_active_return,
        strategy_return - same_exposure_return,
        rtol=0.0,
        atol=1e-12,
    ):
        stop.append("active_return_inconsistent")

    score_psi = observation.get("score_psi", np.nan)
    if np.isfinite(score_psi) and score_psi > spec["monitoring"]["score_population_stability_index_warning"]:
        warnings.append("score_distribution_drift")
    calibration_slope = observation.get("calibration_slope", np.nan)
    lower, upper = spec["monitoring"]["calibration_slope_warning_range"]
    if np.isfinite(calibration_slope) and not lower <= calibration_slope <= upper:
        warnings.append("calibration_slope_warning")

    active_return = supplied_active_return
    prior_active = pd.to_numeric(history.get("active_return"), errors="coerce").dropna()
    active_series = pd.concat([prior_active, pd.Series([active_return])]).dropna()
    active_curve = (1 + active_series).cumprod()
    active_drawdown = (
        float((active_curve / active_curve.cummax() - 1).iloc[-1])
        if not active_curve.empty
        else np.nan
    )
    if (
        np.isfinite(active_drawdown)
        and active_drawdown <= spec["monitoring"]["active_drawdown_vs_same_exposure_stop"]
    ):
        stop.append("active_drawdown_stop")

    status = "stop" if stop else ("warning" if warnings else "pass")
    return {
        **{column: observation.get(column, np.nan) for column in LEDGER_COLUMNS},
        "observation_date": date.date().isoformat(),
        "recorded_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "freeze_hash": spec["freeze_hash"],
        "active_drawdown": active_drawdown,
        "warning_reasons": "|".join(warnings),
        "stop_reasons": "|".join(stop),
        "monitor_status": status,
    }


def append_shadow_observation(spec_path, ledger_path, observation):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    ledger_path = initialize_shadow_ledger(ledger_path)
    history = pd.read_csv(ledger_path)
    date = pd.Timestamp(observation["observation_date"]).date().isoformat()
    if not history.empty and history["observation_date"].astype(str).eq(date).any():
        raise ValueError(f"Shadow observation already recorded: {date}")
    row = monitor_observation(spec, observation, history=history)
    if row["monitor_status"] == "stop":
        raise RuntimeError(f"Shadow observation rejected: {row['stop_reasons']}")
    pd.concat([history, pd.DataFrame([row], columns=LEDGER_COLUMNS)]).to_csv(
        ledger_path, index=False
    )
    return row
