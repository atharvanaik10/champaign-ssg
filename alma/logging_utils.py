from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Initialize a simple, consistent logging format for apps/demos.

    Args:
        level: Root logging level (e.g., logging.INFO, logging.DEBUG).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
