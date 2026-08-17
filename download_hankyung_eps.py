"""Download Hankyung Consensus EPS estimates and calculate their dispersion."""

from __future__ import annotations

import argparse
import os
from datetime import date, timedelta
from pathlib import Path

from src.hankyung_eps import download_eps_dispersion


def _arguments():
    today = date.today()
    parser = argparse.ArgumentParser(
        description=(
            "Download Korean company EPS estimates from Hankyung Consensus and "
            "calculate stock/fiscal-period standard deviations."
        )
    )
    parser.add_argument(
        "--from-date",
        default=(today - timedelta(days=90)).isoformat(),
        help="inclusive report start date (YYYY-MM-DD; default: 90 days ago)",
    )
    parser.add_argument(
        "--to-date",
        default=today.isoformat(),
        help="inclusive report end date (YYYY-MM-DD; default: today)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Data/hankyung_eps_dispersion.csv"),
        help="summary CSV path",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path("Data/hankyung_eps_estimates.csv"),
        help="normalized estimate CSV path",
    )
    parser.add_argument(
        "--dedupe-by",
        choices=["brokerage", "brokerage_analyst", "none"],
        default="brokerage",
        help="retain only the latest estimate per contributor (default: brokerage)",
    )
    parser.add_argument(
        "--min-estimates",
        type=int,
        default=2,
        help="minimum estimates required in a summary row (default: 2)",
    )
    parser.add_argument(
        "--ddof",
        type=int,
        choices=[0, 1],
        default=1,
        help="0=population stddev, 1=sample stddev (default: 1)",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=1000,
        help="reports requested per API page (default/max: 1000)",
    )
    parser.add_argument(
        "--include-zero-eps",
        action="store_true",
        help="include exact-zero EPS values (often API missing-value placeholders)",
    )
    return parser.parse_args()


def main():
    args = _arguments()
    token = os.environ.get("HANKYUNG_API_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "HANKYUNG_API_TOKEN is not set. Set it in the current process and retry."
        )

    estimates, dispersion = download_eps_dispersion(
        token,
        args.from_date,
        args.to_date,
        dedupe_by=args.dedupe_by,
        min_estimates=args.min_estimates,
        ddof=args.ddof,
        exclude_zero=not args.include_zero_eps,
        page_size=args.page_size,
    )
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(args.raw_output, index=False, encoding="utf-8-sig")
    dispersion.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(
        f"Saved {len(estimates):,} normalized estimates to {args.raw_output.resolve()}"
    )
    print(
        f"Saved {len(dispersion):,} stock-period summaries to {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
