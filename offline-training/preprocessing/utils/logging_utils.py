from __future__ import annotations
import logging
from typing import Optional


def get_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Tạo logger đơn giản cho toàn bộ preprocessing.

    - Format: [LEVEL] name - message
    - Không cấu hình file handler ở đây (giữ đơn giản, dễ import).
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
