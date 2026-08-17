"""Build the point-in-time marcap KOSPI top-200 price proxy cache."""

from src.config import (
    KOREA_STOCK_PANEL_FILE,
    MARCAP_KOSPI200_FILE,
    MARCAP_KOSPI200_METADATA_FILE,
)
from src.marcap_kospi200 import build_marcap_kospi200_proxy


if __name__ == "__main__":
    frame, metadata = build_marcap_kospi200_proxy(
        KOREA_STOCK_PANEL_FILE,
        MARCAP_KOSPI200_FILE,
        MARCAP_KOSPI200_METADATA_FILE,
    )
    print(
        f"{metadata['proxy_name']}: {len(frame):,} months, "
        f"through {metadata['complete_through']}"
    )
