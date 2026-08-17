"""Score a frozen monthly EWS strategy without repeating model selection.

The frozen specification fixes the feature names, model hyperparameters,
training purge, random seed, allocation policy, and portfolio conventions.
Only the expanding training sample and the new month of input data may change.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.backtest import annual_yield_to_monthly_return
from src.modeling import fit_latest_ews
from src.position_sizing import target_weight_from_ews
from src.shadow import monitor_observation, validate_frozen_spec


SCORING_PROTOCOL = "monthly_expanding_refit_v1"


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path(run_dir: Path, relative_name: str) -> Path:
    run_dir = run_dir.resolve()
    path = (run_dir / relative_name).resolve()
    try:
        path.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError(f"Artifact must stay inside run directory: {relative_name}") from exc
    return path


def _read_indexed_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.DatetimeIndex(frame.index).to_period("M").to_timestamp("M")
    frame = frame.sort_index()
    if frame.index.duplicated().any():
        raise ValueError(f"Duplicate month in {path.name}")
    return frame


def load_frozen_scoring_spec(spec_path: str | Path) -> dict:
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    validate_frozen_spec(spec)
    required = {
        "scoring_protocol",
        "forecast_horizon_months",
        "minimum_training_months",
        "random_state",
        "model_params",
        "sizing_config",
        "score_history_file",
        "score_history_sha256",
        "signal_market_file",
        "signal_market_column",
        "portfolio_price_file",
        "portfolio_price_column",
        "cash_yield_file",
        "cash_yield_column",
        "cash_return_convention",
        "same_exposure_stock_weight",
    }
    missing = required.difference(spec)
    if missing:
        raise ValueError(
            "Frozen scoring spec missing fields: " + ", ".join(sorted(missing))
        )
    if spec["scoring_protocol"] != SCORING_PROTOCOL:
        raise ValueError(f"Unsupported scoring protocol: {spec['scoring_protocol']}")
    if spec.get("model") != "mlp":
        raise ValueError("This scorer requires a frozen MLP specification")
    return spec


def _require_complete_run(run_dir: Path) -> dict:
    manifest_path = run_dir / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing experiment manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Data run is not complete: {manifest.get('status')}")
    return manifest


def _load_baseline_scores(spec_run_dir: Path, spec: dict) -> pd.Series:
    path = _artifact_path(spec_run_dir, spec["score_history_file"])
    actual_hash = _sha256_file(path)
    if actual_hash != spec["score_history_sha256"]:
        raise ValueError("Frozen score-history artifact hash mismatch")
    frame = _read_indexed_csv(path)
    if "raw_ews" not in frame:
        raise ValueError("Frozen score history is missing raw_ews")
    scores = pd.to_numeric(frame["raw_ews"], errors="coerce").dropna()
    freeze_date = pd.Timestamp(spec["freeze_date"])
    if scores.empty or scores.index.max() != freeze_date:
        raise ValueError("Frozen score history must end exactly on freeze_date")
    if not 0.0 <= scores.min() or scores.max() > 100.0:
        raise ValueError("Frozen score history contains a value outside 0..100")
    return scores.rename("raw_ews")


def _load_prior_shadow_scores(
    spec_run_dir: Path,
    spec: dict,
    *,
    asof_date: pd.Timestamp,
) -> tuple[pd.Series, pd.DataFrame]:
    ledger_path = _artifact_path(
        spec_run_dir, spec.get("ledger_file", "mlp_research_shadow_ledger.csv")
    )
    if not ledger_path.exists():
        return pd.Series(dtype=float, name="raw_ews"), pd.DataFrame()
    ledger = pd.read_csv(ledger_path)
    if ledger.empty:
        return pd.Series(dtype=float, name="raw_ews"), ledger
    dates = pd.to_datetime(ledger["observation_date"]).dt.to_period("M").dt.to_timestamp("M")
    if dates.duplicated().any():
        raise ValueError("Shadow ledger contains duplicate observation months")
    if not ledger["freeze_hash"].astype(str).eq(spec["freeze_hash"]).all():
        raise ValueError("Shadow ledger contains a different freeze hash")
    if "monitor_status" in ledger and ledger["monitor_status"].astype(str).eq("stop").any():
        raise ValueError("Shadow ledger contains a stopped observation")
    ledger = ledger.copy()
    ledger.index = pd.DatetimeIndex(dates)
    if asof_date in ledger.index:
        raise ValueError(f"Shadow month is already recorded: {asof_date.date()}")
    prior = ledger.loc[ledger.index < asof_date]
    scores = pd.to_numeric(prior.get("raw_ews"), errors="coerce").dropna()
    scores.name = "raw_ews"
    return scores, ledger


def _require_contiguous_prior_months(
    spec: dict,
    prior_shadow_scores: pd.Series,
    asof_date: pd.Timestamp,
) -> None:
    first = pd.Timestamp(spec["first_eligible_observation"])
    prior_end = asof_date - pd.offsets.MonthEnd(1)
    if prior_end < first:
        return
    expected = pd.date_range(first, prior_end, freq="ME")
    missing = expected.difference(prior_shadow_scores.index)
    if len(missing):
        raise ValueError(
            "Missing prior frozen shadow score for "
            + ", ".join(date.date().isoformat() for date in missing)
        )


def _resolve_asof_date(
    data_run_dir: Path,
    spec: dict,
    requested: str | pd.Timestamp | None,
) -> pd.Timestamp:
    signal_path = _artifact_path(data_run_dir, spec["signal_market_file"])
    signal = _read_indexed_csv(signal_path)
    column = spec["signal_market_column"]
    if column not in signal:
        raise ValueError(f"Signal market column is missing: {column}")
    completed = pd.to_numeric(signal[column], errors="coerce").dropna()
    if completed.empty:
        raise ValueError("Signal market has no completed month")
    if requested is None:
        return completed.index.max()
    asof = pd.Timestamp(requested)
    month_end = asof.to_period("M").to_timestamp("M")
    if asof.normalize() != month_end:
        raise ValueError("asof_date must be a calendar month-end")
    if month_end not in completed.index:
        raise ValueError(f"Signal market month is not complete: {month_end.date()}")
    return month_end


def _realized_portfolio_returns(
    data_run_dir: Path,
    spec: dict,
    *,
    asof_date: pd.Timestamp,
    executed_weight: float,
    turnover: float,
) -> dict:
    price_path = _artifact_path(data_run_dir, spec["portfolio_price_file"])
    price_frame = _read_indexed_csv(price_path)
    price_column = spec["portfolio_price_column"]
    if price_column not in price_frame:
        raise ValueError(f"Portfolio price column is missing: {price_column}")
    price = pd.to_numeric(price_frame[price_column], errors="coerce").dropna()
    previous_month = asof_date - pd.offsets.MonthEnd(1)
    if asof_date not in price.index or previous_month not in price.index:
        raise ValueError("Portfolio price is missing the current or prior month")
    market_return = float(price.loc[asof_date] / price.loc[previous_month] - 1.0)

    cash_path = _artifact_path(data_run_dir, spec["cash_yield_file"])
    if cash_path.suffix.lower() == ".parquet":
        cash_frame = pd.read_parquet(cash_path)
        cash_frame.index = (
            pd.DatetimeIndex(cash_frame.index).to_period("M").to_timestamp("M")
        )
    else:
        cash_frame = _read_indexed_csv(cash_path)
    cash_column = spec["cash_yield_column"]
    if cash_column not in cash_frame:
        raise ValueError(f"Cash-yield column is missing: {cash_column}")
    cash_yield = pd.to_numeric(cash_frame[cash_column], errors="coerce").sort_index()
    prior_cash = cash_yield.loc[:previous_month].dropna()
    if prior_cash.empty or prior_cash.index.max() != previous_month:
        raise ValueError("Prior-month cash yield is unavailable")
    cash_return = float(
        annual_yield_to_monthly_return(
            pd.Series([prior_cash.iloc[-1]]),
            convention=spec["cash_return_convention"],
        ).iloc[0]
    )
    transaction_cost = float(turnover) * float(spec["transaction_cost_bps"]) / 10000.0
    strategy_return = (
        float(executed_weight) * market_return
        + (1.0 - float(executed_weight)) * cash_return
        - transaction_cost
    )
    same_weight = float(spec["same_exposure_stock_weight"])
    same_exposure_return = (
        same_weight * market_return + (1.0 - same_weight) * cash_return
    )
    return {
        "market_return": market_return,
        "cash_return": cash_return,
        "transaction_cost": transaction_cost,
        "strategy_return": float(strategy_return),
        "same_exposure_return": float(same_exposure_return),
        "active_return": float(strategy_return - same_exposure_return),
    }


def score_frozen_mlp(
    spec_run_dir: str | Path,
    data_run_dir: str | Path | None = None,
    *,
    spec_name: str = "mlp_research_shadow_spec.json",
    asof_date: str | pd.Timestamp | None = None,
) -> dict:
    """Return one auditable signal/return packet for a frozen MLP strategy."""

    spec_run_dir = Path(spec_run_dir).resolve()
    data_run_dir = (
        spec_run_dir if data_run_dir is None else Path(data_run_dir).resolve()
    )
    _require_complete_run(spec_run_dir)
    data_manifest = _require_complete_run(data_run_dir)
    spec_path = _artifact_path(spec_run_dir, spec_name)
    spec = load_frozen_scoring_spec(spec_path)
    asof = _resolve_asof_date(data_run_dir, spec, asof_date)
    freeze_date = pd.Timestamp(spec["freeze_date"])
    if asof < freeze_date:
        raise ValueError("Frozen scorer cannot evaluate a month before freeze_date")

    factor_path = _artifact_path(data_run_dir, spec.get("factor_matrix_file", "factor_matrix.parquet"))
    target_path = _artifact_path(data_run_dir, spec.get("target_file", "target.csv"))
    factors = pd.read_parquet(factor_path)
    factors.index = pd.DatetimeIndex(factors.index).to_period("M").to_timestamp("M")
    target = _read_indexed_csv(target_path)
    if "y" not in target:
        raise ValueError("Target artifact is missing y")
    missing_features = [feature for feature in spec["features"] if feature not in factors]
    if missing_features:
        raise ValueError("Data run is missing frozen features: " + ", ".join(missing_features))
    if asof not in factors.index:
        raise ValueError(f"Factor matrix is missing as-of month: {asof.date()}")
    feature_row = factors.loc[asof, spec["features"]]
    if feature_row.isna().any():
        missing = feature_row.index[feature_row.isna()].tolist()
        raise ValueError("As-of frozen feature is missing: " + ", ".join(missing))

    with contextlib.redirect_stdout(io.StringIO()):
        latest = fit_latest_ews(
            X=factors,
            y=target["y"],
            features=spec["features"],
            horizon=int(spec["forecast_horizon_months"]),
            asof_date=asof,
            min_train=int(spec["minimum_training_months"]),
            model_type="mlp",
            mlp_params=spec["model_params"],
            random_state=int(spec["random_state"]),
        )
    if pd.Timestamp(latest["date"]) != asof:
        raise ValueError("Frozen scorer silently fell back to an earlier feature month")
    label_cutoff = (
        asof.to_period("M") - int(spec["forecast_horizon_months"])
    ).to_timestamp("M")
    if pd.Timestamp(latest["train_end"]) > label_cutoff:
        raise RuntimeError("Training data exceeded the purged label cutoff")

    baseline_scores = _load_baseline_scores(spec_run_dir, spec)
    prior_shadow_scores, ledger = _load_prior_shadow_scores(
        spec_run_dir, spec, asof_date=asof
    )
    if asof > freeze_date:
        _require_contiguous_prior_months(spec, prior_shadow_scores, asof)
    score_history = pd.concat(
        [
            baseline_scores,
            prior_shadow_scores,
            pd.Series({asof: float(latest["ews"])}, name="raw_ews"),
        ]
    ).sort_index()
    score_history = score_history[~score_history.index.duplicated(keep="last")]
    targets = target_weight_from_ews(
        score_history,
        policy=spec["allocation_policy"],
        **spec["sizing_config"],
    )
    previous_month = asof - pd.offsets.MonthEnd(1)
    two_months_prior = asof - pd.offsets.MonthEnd(2)
    required_weight_months = [previous_month, two_months_prior]
    missing_weight_months = [date for date in required_weight_months if date not in targets.index]
    if missing_weight_months:
        raise ValueError(
            "Cannot reconstruct one-month execution history: "
            + ", ".join(date.date().isoformat() for date in missing_weight_months)
        )
    target_weight = float(targets.loc[asof])
    executed_weight = float(targets.loc[previous_month])
    prior_executed_weight = float(targets.loc[two_months_prior])
    if not all(np.isfinite(value) for value in (target_weight, executed_weight, prior_executed_weight)):
        raise ValueError("Frozen allocation policy produced a missing weight")
    turnover = abs(executed_weight - prior_executed_weight)
    realized = _realized_portfolio_returns(
        data_run_dir,
        spec,
        asof_date=asof,
        executed_weight=executed_weight,
        turnover=turnover,
    )

    packet = {
        "packet_schema_version": 1,
        "packet_type": "frozen_mlp_monthly_observation",
        "observation_date": asof.date().isoformat(),
        "generated_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "source_spec_run": spec_run_dir.name,
        "source_data_run": data_manifest.get("run_id", data_run_dir.name),
        "freeze_hash": spec["freeze_hash"],
        "capital_authorized": bool(spec.get("capital_authorized", False)),
        "feature_values": {
            feature: float(feature_row.loc[feature]) for feature in spec["features"]
        },
        "raw_ews": float(latest["ews"]),
        "target_stock_weight": target_weight,
        "executed_stock_weight": executed_weight,
        "prior_executed_stock_weight": prior_executed_weight,
        "turnover": float(turnover),
        **realized,
        "model": "mlp",
        "allocation_policy": spec["allocation_policy"],
        "training_observations": int(latest["train_n"]),
        "training_start": pd.Timestamp(latest["train_start"]).date().isoformat(),
        "training_end": pd.Timestamp(latest["train_end"]).date().isoformat(),
        "label_cutoff_date": label_cutoff.date().isoformat(),
    }
    if asof == freeze_date:
        packet.update(
            {
                "packet_status": "freeze_reproduction_diagnostic_only",
                "appendable_to_shadow_ledger": False,
                "monitor_status": "not_applicable_before_first_eligible_observation",
            }
        )
        return packet

    preview = monitor_observation(spec, packet, history=ledger)
    if preview["monitor_status"] == "stop":
        raise RuntimeError(f"Frozen observation rejected: {preview['stop_reasons']}")
    packet.update(
        {
            "packet_status": "ready_for_shadow_append",
            "appendable_to_shadow_ledger": True,
            "monitor_status": preview["monitor_status"],
            "warning_reasons": preview["warning_reasons"],
            "active_drawdown": preview["active_drawdown"],
        }
    )
    return packet
