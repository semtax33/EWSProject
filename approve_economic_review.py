"""Approve an explicitly completed human economic-review checklist."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import ECONOMIC_REVIEW_FILE
from src.economic_review import merge_completed_review, write_registry_atomic


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate and register a human-completed EWS economic review. "
            "This command never fills or approves fields automatically."
        )
    )
    parser.add_argument("--completed-review", required=True)
    parser.add_argument("--registry", default=str(ECONOMIC_REVIEW_FILE))
    return parser.parse_args()


def main():
    args = parse_args()
    review_path = Path(args.completed_review).resolve()
    registry_path = Path(args.registry).resolve()
    completed = pd.read_csv(review_path)
    registry = pd.read_csv(registry_path)
    merged = merge_completed_review(registry, completed)
    write_registry_atomic(merged, registry_path)
    print(
        f"Approved {len(completed)} feature reviews in {registry_path}. "
        "Rerun the full pipeline; approval does not bypass other gates."
    )


if __name__ == "__main__":
    main()
