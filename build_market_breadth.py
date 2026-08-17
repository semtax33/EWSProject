"""Build cached market-breadth candidates from the large stock panel."""

import argparse

from src.config import (
    KOREA_STOCK_PANEL_FILE,
    MARKET_BREADTH_FILE,
    MARKET_BREADTH_METADATA_FILE,
)
from src.market_breadth import build_market_breadth


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(KOREA_STOCK_PANEL_FILE))
    parser.add_argument("--output", default=str(MARKET_BREADTH_FILE))
    parser.add_argument("--metadata", default=str(MARKET_BREADTH_METADATA_FILE))
    parser.add_argument("--chunksize", type=int, default=500_000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    factors, metadata = build_market_breadth(
        args.source,
        args.output,
        args.metadata,
        chunksize=args.chunksize,
    )
    print(f"saved {factors.shape[0]:,} x {factors.shape[1]:,} market-breadth matrix")
    print(metadata)
