"""Evaluate a common linear-backbone/MLP-confirmation protocol before 2020.

The nonlinear model is allowed to adjust confidence only when its directional
classification agrees with the regularized Logistic backbone.  On disagreement
the backbone is retained, so a small-sample MLP cannot reverse a more stable
linear relation.  The protocol first prefers a fully autonomous MLP; otherwise
it selects the strongest guarded MLP correction that clears both signal and
portfolio gates.  No 2020+ observation is used in this selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_anchored_mlp_preholdout_research import (
    evaluate_candidate,
    load_market_inputs,
)


GUARDED_STRENGTHS = (1.00, 0.50, 0.25)
RISK_VETO_THRESHOLDS = (0.50, 0.35)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _load_endpoint_predictions(prediction_dir: Path, market_key: str):
    mlp = pd.read_csv(
        prediction_dir / f"{market_key}_anchor_00_prediction.csv",
        index_col=0,
        parse_dates=True,
    ).iloc[:, 0]
    logistic = pd.read_csv(
        prediction_dir / f"{market_key}_anchor_100_prediction.csv",
        index_col=0,
        parse_dates=True,
    ).iloc[:, 0]
    signal_folds = pd.read_csv(
        prediction_dir / f"{market_key}_anchor_00_signal_folds.csv",
        parse_dates=["model_eligibility_start"],
    )
    selections = signal_folds[["fold", "model_eligibility_start"]].copy()
    return mlp, logistic, selections


def guarded_prediction(logistic, mlp, strength):
    """Return causal predictions with MLP confidence updates on agreement only."""
    common = logistic.dropna().index.intersection(mlp.dropna().index)
    result = pd.Series(float("nan"), index=logistic.index, dtype=float)
    base = logistic.loc[common]
    deep = mlp.loc[common]
    agreement = base.ge(0.5).eq(deep.ge(0.5))
    result.loc[common] = base
    result.loc[common[agreement]] = (
        (1.0 - strength) * base.loc[agreement]
        + strength * deep.loc[agreement]
    )
    return result, agreement


def risk_veto_prediction(logistic, mlp, mlp_safe_probability_threshold):
    """Let the MLP veto an aggressive linear risk-on score, never create one."""
    common = logistic.dropna().index.intersection(mlp.dropna().index)
    result = pd.Series(float("nan"), index=logistic.index, dtype=float)
    base = logistic.loc[common]
    deep = mlp.loc[common]
    veto = base.ge(0.65) & deep.lt(mlp_safe_probability_threshold)
    result.loc[common] = base
    # A veto moves the score to neutral; it does not make an outright short call.
    result.loc[veto.index[veto]] = 0.50
    return result, veto


def run(run_dirs, prediction_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in map(Path, run_dirs):
        inputs = load_market_inputs(run_dir.resolve())
        market_key = inputs["market_key"]
        print(f"[GUARDED MLP PRE-2020] {inputs['market_name']}", flush=True)
        mlp, logistic, selections = _load_endpoint_predictions(
            prediction_dir, market_key
        )
        completed = set(selections["fold"])
        folds = [fold for fold in inputs["folds"] if fold.fold in completed]

        candidates = [
            (
                "pure_mlp",
                1.0,
                mlp,
                pd.Series(True, index=mlp.dropna().index),
                "autonomous",
            )
        ]
        for strength in GUARDED_STRENGTHS:
            prediction, agreement = guarded_prediction(logistic, mlp, strength)
            candidates.append(
                (
                    f"guarded_{strength:.2f}",
                    strength,
                    prediction,
                    ~agreement,
                    "agreement_correction",
                )
            )
        for threshold in RISK_VETO_THRESHOLDS:
            prediction, veto = risk_veto_prediction(logistic, mlp, threshold)
            candidates.append(
                (
                    f"risk_veto_{threshold:.2f}",
                    1.0,
                    prediction,
                    veto,
                    "risk_veto",
                )
            )

        for candidate, strength, prediction, intervention, role in candidates:
            result = evaluate_candidate(
                inputs, prediction, selections, folds, anchor_weight=0.0
            )
            row, comparison, monthly, fold_results, decisions, fold_rows = result
            row.pop("anchor_weight")
            row.pop("mlp_weight")
            row.update(
                {
                    "candidate": candidate,
                    "mlp_correction_strength": strength,
                    "linear_backbone": candidate != "pure_mlp",
                    "mlp_role": role,
                    "mlp_intervention_ratio": float(intervention.mean()),
                    "historical_holdout_opened": False,
                }
            )
            rows.append(row)
            stem = output_dir / f"{market_key}_{candidate}"
            prediction.to_csv(f"{stem}_prediction.csv")
            comparison.to_csv(f"{stem}_policy_comparison.csv", index=False)
            monthly.to_csv(f"{stem}_policy_monthly.csv")
            fold_results.to_csv(f"{stem}_policy_folds.csv", index=False)
            decisions.to_csv(f"{stem}_policy_gate.csv", index=False)
            fold_rows.to_csv(f"{stem}_signal_folds.csv", index=False)

    summary = pd.DataFrame(rows)
    selected = {}
    for market_key, market_rows in summary.groupby("market_key"):
        pure = market_rows.loc[
            market_rows["candidate"].eq("pure_mlp") & market_rows["joint_gate"]
        ]
        guarded = market_rows.loc[
            ~market_rows["candidate"].eq("pure_mlp") & market_rows["joint_gate"]
        ].sort_values(
            ["mlp_intervention_ratio", "mlp_correction_strength"],
            ascending=False,
        )
        if not pure.empty:
            choice = pure.iloc[0]
        elif not guarded.empty:
            choice = guarded.iloc[0]
        else:
            selected[market_key] = None
            continue
        selected[market_key] = choice["candidate"]
    summary["selected_market_candidate"] = summary.apply(
        lambda row: row["candidate"] == selected[row["market_key"]], axis=1
    )
    summary.to_csv(output_dir / "cross_market_guarded_mlp_summary.csv", index=False)
    (output_dir / "selection.json").write_text(
        json.dumps(
            {
                "selected_candidate_by_market": selected,
                "selection_scope": "pre2020_all_markets_only",
                "selection_rule": (
                    "pure_mlp_if_joint_gate_else_most_active_guarded_mlp_that_passes"
                ),
                "guarded_strengths": list(GUARDED_STRENGTHS),
                "risk_veto_thresholds": list(RISK_VETO_THRESHOLDS),
                "historical_holdout_opened": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(json.dumps(selected, indent=2))


def main():
    args = parse_args()
    run(
        args.run_dir,
        Path(args.prediction_dir).resolve(),
        Path(args.output_dir).resolve(),
    )


if __name__ == "__main__":
    main()
