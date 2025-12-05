from __future__ import annotations
from dataclasses import dataclass, field
import time
from typing import Optional

from .logging_utils import get_logger


@dataclass
class Timer:
    """
    Context manager đo thời gian chạy một khối code.

    Ví dụ:
        from preprocessing.utils import Timer

        with Timer("preprocess_metadata"):
            run_preprocess_metadata()

    Output:
        [INFO] offline-training - [TIMER] preprocess_metadata took 3.421s
    """
    name: str = "block"
    logger_name: Optional[str] = None
    _start: float = field(init=False, default=0.0)

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self._start
        logger = get_logger(self.logger_name)
        logger.info(f"[TIMER] {self.name} took {elapsed:.3f}s")
        # Không chặn exception
        return False
