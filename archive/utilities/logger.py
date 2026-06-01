"""
utilities/logger.py
Centralised logging — writes to logs/app.log and console.
"""
import logging
import os
from core.config import get_settings


def get_logger(name: str) -> logging.Logger:
    settings = get_settings()
    os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(settings.log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger
