from __future__ import annotations
from typing import Dict, Any, Optional
import math

import numpy as np

from .text_utils import LAUGH_EMOJIS


def parse_count(value: Optional[str]) -> Optional[float]:
    """Parse các trường likes/shares/... dạng '307K', '4.5M', '3537', 'N/A'."""
    if value is None:
        return None
    # If list, take length? Or if it's data inconsistency. 
    # For tiktok-data, "comments" might be the list of comments itself?
    # If list, return len(list)
    if isinstance(value, list):
        return float(len(value))
    
    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip()
    if value == "" or value.upper() == "N/A":
        return None

    v = value.upper()
    try:
        if v.endswith("K"):
            return float(v[:-1]) * 1e3
        if v.endswith("M"):
            return float(v[:-1]) * 1e6
        if v.endswith("B"):
            return float(v[:-1]) * 1e9
        return float(v)
    except ValueError:
        return None


def safe_log1p(x: Optional[float]) -> float:
    if x is None or x < 0:
        return 0.0
    return math.log1p(x)


def compute_numeric_features(meta: Dict[str, Any]) -> Dict[str, float]:
    likes = parse_count(meta.get("likes"))
    comments = parse_count(meta.get("comments"))
    shares = parse_count(meta.get("shares"))
    bookmarks = parse_count(meta.get("bookmarks"))
    views = parse_count(meta.get("views"))

    likes_log = safe_log1p(likes)
    comments_log = safe_log1p(comments)
    shares_log = safe_log1p(shares)
    bookmarks_log = safe_log1p(bookmarks)
    views_log = safe_log1p(views)

    def ratio(a: Optional[float], b: Optional[float]) -> float:
        if a is None or b is None or b <= 0:
            return 0.0
        return float(a) / float(b)

    like_rate = ratio(likes, views)
    comment_rate = ratio(comments, views)
    share_rate = ratio(shares, views)
    bookmark_rate = ratio(bookmarks, views)
    engagement_rate = ratio(
        (likes or 0) + (comments or 0) + (shares or 0) + (bookmarks or 0),
        views,
    )

    out = {
        "likes_log": likes_log,
        "comments_log": comments_log,
        "shares_log": shares_log,
        "bookmarks_log": bookmarks_log,
        "views_log": views_log,
        "like_rate": like_rate,
        "comment_rate": comment_rate,
        "share_rate": share_rate,
        "bookmark_rate": bookmark_rate,
        "engagement_rate": engagement_rate,
        "has_views": 1.0 if views is not None else 0.0,
        "has_bookmarks": 1.0 if bookmarks is not None else 0.0,
    }
    return out
