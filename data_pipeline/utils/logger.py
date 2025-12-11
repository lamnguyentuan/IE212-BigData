"""
Logger Utility for Data Pipeline.

Provides a standardized logger with consistent check for handlers to avoid duplication.

Usage:
    from data_pipeline.utils.logger import get_logger
    logger = get_logger("my_module")
    logger.info("Hello world")
"""

import logging
import sys
from typing import Optional

def get_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Get a configured logger instance.
    If name is None, returns root logger 'data-pipeline'.
    """
    logger_name = name or "data-pipeline"
    logger = logging.getLogger(logger_name)
    
    # If logger already has handlers, assume it was configured (unless we force reconfig)
    # This prevents duplicate logs if get_logger is called multiple times.
    if logger.handlers:
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    logger.propagate = False
    
    return logger
