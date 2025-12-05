from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Set

LAUGH_EMOJIS: Set[str] = {"😂", "🤣", "😄", "😆", "😅", "😹", "😸"}
SAD_EMOJIS: Set[str] = {"😔", "😢", "😭", "😞", "☹️", "🙁"}
SCARE_EMOJIS: Set[str] = {"😱", "🤯", "😨", "😰", "😧"}


def map_emojis_to_tokens(text: str) -> str:
    """Map emoji → token đặc biệt để giữ lại signal trong text."""
    for e in LAUGH_EMOJIS:
        text = text.replace(e, " <EMOJI_LAUGH> ")
    for e in SAD_EMOJIS:
        text = text.replace(e, " <EMOJI_SAD> ")
    for e in SCARE_EMOJIS:
        text = text.replace(e, " <EMOJI_SCARE> ")
    return text


def strip_hashtags(text: str) -> str:
    """Xoá hashtag khỏi description (tags đã có field riêng)."""
    return re.sub(r"#\S+", "", text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
