from __future__ import annotations
from typing import List, Dict, Any

import numpy as np

from .text_utils import LAUGH_EMOJIS


def flatten_comments_tree(comments_tree: List[Dict[str, Any]]) -> List[str]:
    texts: List[str] = []
    for c in comments_tree:
        main_text = c.get("text", "")
        if main_text:
            texts.append(main_text)
        for r in c.get("replies", []):
            reply_text = r.get("text", "")
            if reply_text:
                texts.append(reply_text)
    return texts


def compute_comment_basic_features(flat_comments: List[str]) -> Dict[str, float]:
    n_comments = len(flat_comments)
    if n_comments == 0:
        return {
            "n_comments": 0.0,
            "avg_len_chars": 0.0,
            "avg_len_words": 0.0,
            "frac_has_laugh_emoji": 0.0,
            "frac_has_question": 0.0,
        }

    lengths_chars = [len(c) for c in flat_comments]
    lengths_words = [len(c.split()) for c in flat_comments]

    def has_any_emoji(text: str, emojis: set[str]) -> bool:
        return any(e in text for e in emojis)

    has_laugh = [has_any_emoji(c, LAUGH_EMOJIS) for c in flat_comments]
    has_question = ["?" in c for c in flat_comments]

    avg_len_chars = float(np.mean(lengths_chars))
    avg_len_words = float(np.mean(lengths_words))
    frac_laugh = float(np.mean(has_laugh))
    frac_question = float(np.mean(has_question))

    return {
        "n_comments": float(n_comments),
        "avg_len_chars": avg_len_chars,
        "avg_len_words": avg_len_words,
        "frac_has_laugh_emoji": frac_laugh,
        "frac_has_question": frac_question,
    }
