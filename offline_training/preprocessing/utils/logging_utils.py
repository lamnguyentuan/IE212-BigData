from __future__ import annotations
import logging
from typing import Optional


def get_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Get a configured logger wrapper for preprocessing.

    Usage:
        logger = get_logger("preprocess")

    Args:
        name: Logger name (default: offline-training)
        level: Logging level (default: INFO)
    """
    logger_name = name or "offline-training"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        # Đã cấu hình từ trước → chỉ set level rồi trả về
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(levelname)s] %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
