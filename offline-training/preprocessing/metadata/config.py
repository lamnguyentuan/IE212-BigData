from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import datetime as dt
import yaml
from pathlib import Path


@dataclass
class MetadataConfig:
    text_model_name: str = "vinai/phobert-base-v2"
    toxicity_model_name: str = "funa21/phobert-finetuned-victsd-toxic-v2"
    construct_model_name: str = "funa21/phobert-finetuned-victsd-constructiveness-v2"
    norm_model_name: str = "meoo225/ViNormT5"

    reference_date: dt.date = dt.date(2025, 12, 5)
    assume_year: int = 2025

    max_desc_len: int = 128
    max_tags_len: int = 64
    max_comments_len: int = 256


def load_metadata_config(path: str | Path) -> MetadataConfig:
    path = Path(path)
    if not path.exists():
        return MetadataConfig()

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # parse date nếu có
    ref_date_str = raw.get("reference_date")
    reference_date = MetadataConfig.reference_date
    if isinstance(ref_date_str, str):
        # format: YYYY-MM-DD
        reference_date = dt.date.fromisoformat(ref_date_str)

    return MetadataConfig(
        text_model_name=raw.get("text_model_name", MetadataConfig.text_model_name),
        toxicity_model_name=raw.get(
            "toxicity_model_name", MetadataConfig.toxicity_model_name
        ),
        construct_model_name=raw.get(
            "construct_model_name", MetadataConfig.construct_model_name
        ),
        norm_model_name=raw.get("norm_model_name", MetadataConfig.norm_model_name),
        reference_date=reference_date,
        assume_year=raw.get("assume_year", MetadataConfig.assume_year),
        max_desc_len=raw.get("max_desc_len", MetadataConfig.max_desc_len),
        max_tags_len=raw.get("max_tags_len", MetadataConfig.max_tags_len),
        max_comments_len=raw.get("max_comments_len", MetadataConfig.max_comments_len),
    )
