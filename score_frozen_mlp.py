"""Generate one monthly observation packet from a frozen MLP specification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.frozen_scoring import score_frozen_mlp


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply frozen MLP features, hyperparameters, purge and allocation "
            "rules to a completed data run without repeating feature selection."
        )
    )
    parser.add_argument("--spec-run-dir", required=True)
    parser.add_argument(
        "--data-run-dir",
        help="completed run with new factor/target/market artifacts; defaults to spec run",
    )
    parser.add_argument(
        "--spec-name",
        default="mlp_research_shadow_spec.json",
    )
    parser.add_argument(
        "--asof-date",
        help="calendar month-end; defaults to latest completed selected-market month",
    )
    parser.add_argument(
        "--output",
        help="optional JSON output path; stdout is always printed",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    packet = score_frozen_mlp(
        args.spec_run_dir,
        args.data_run_dir,
        spec_name=args.spec_name,
        asof_date=args.asof_date,
    )
    encoded = json.dumps(packet, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
