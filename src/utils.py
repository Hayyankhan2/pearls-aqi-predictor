"""Small shared utilities: logging + time helpers."""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone

import pandas as pd


def get_logger(name: str) -> logging.Logger:
    """Consistent, pipeline-friendly logger (stdout, so GitHub Actions captures it)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_date(value) -> pd.Timestamp:
    """Normalise anything date-like to a midnight-floored pandas Timestamp."""
    return pd.to_datetime(value).normalize()
