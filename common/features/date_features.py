from __future__ import annotations
from typing import Optional, Tuple, Dict
import datetime as dt
import math
import re


def parse_mm_dd(date_str: str) -> Optional[Tuple[int, int]]:
    """Parse '8-9' -> (8, 9)."""
    if not date_str:
        return None
    parts = re.split(r"[-/\.]", date_str.strip())
    if len(parts) != 2:
        return None
    try:
        month = int(parts[0])
        day = int(parts[1])
        if 1 <= month <= 12 and 1 <= day <= 31:
            return month, day
    except ValueError:
        return None
    return None


def compute_date_features(
    date_str: Optional[str],
    reference_date: dt.date,
    assume_year: int,
) -> Dict[str, float]:
    """Chuyển date -> age_days + encoding tháng (sin/cos)."""
    if date_str is None:
        return {
            "age_days": 0.0,
            "month_sin": 0.0,
            "month_cos": 0.0,
            "has_date": 0.0,
        }

    md = parse_mm_dd(date_str)
    if md is None:
        return {
            "age_days": 0.0,
            "month_sin": 0.0,
            "month_cos": 0.0,
            "has_date": 0.0,
        }

    month, day = md
    try:
        upload_date = dt.date(assume_year, month, day)
    except ValueError:
        return {
            "age_days": 0.0,
            "month_sin": 0.0,
            "month_cos": 0.0,
            "has_date": 0.0,
        }

    age_days = (reference_date - upload_date).days
    age_days = float(max(age_days, 0))

    month_sin = math.sin(2 * math.pi * month / 12.0)
    month_cos = math.cos(2 * math.pi * month / 12.0)

    return {
        "age_days": age_days,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "has_date": 1.0,
    }
