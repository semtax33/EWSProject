"""Validate and append one post-freeze forward-shadow observation."""

import argparse
import json
from pathlib import Path

from src.shadow import append_shadow_observation, validate_frozen_spec


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--observation-json")
    parser.add_argument(
        "--spec-name",
        default="forward_shadow_spec.json",
        help="frozen spec filename inside run-dir (for example mlp_research_shadow_spec.json)",
    )
    parser.add_argument(
        "--ledger-name",
        help="optional ledger filename; defaults to the spec's ledger_file field",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_dir = Path(args.run_dir)
    spec_path = run_dir / args.spec_name
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    validate_frozen_spec(spec)
    ledger_name = args.ledger_name or spec.get(
        "ledger_file", "forward_shadow_ledger.csv"
    )
    ledger_path = run_dir / ledger_name
    if args.observation_json:
        observation = json.loads(Path(args.observation_json).read_text(encoding="utf-8"))
        row = append_shadow_observation(
            spec_path, ledger_path, observation
        )
        print(json.dumps(row, ensure_ascii=False, indent=2, default=str))
    else:
        print(json.dumps(
            {
                "spec_valid": True,
                "status": spec["status"],
                "freeze_hash": spec["freeze_hash"],
                "capital_authorized": spec.get("capital_authorized"),
                "ledger": str(ledger_path),
            },
            ensure_ascii=False,
            indent=2,
        ))
