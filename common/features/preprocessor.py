from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tqdm import tqdm

from .config import MetadataConfig
from .embeddings import TextNormalizer, HFTextEncoder
from .numeric_features import compute_numeric_features
from .date_features import compute_date_features
from .comments import (
    flatten_comments_tree,
    compute_comment_basic_features,
)
from .classifiers import BinaryCommentClassifier
from .text_utils import strip_hashtags, normalize_whitespace, map_emojis_to_tokens


@dataclass
class MetadataPreprocessor:
    config: MetadataConfig = field(default_factory=MetadataConfig)

    def __post_init__(self) -> None:
        # text normalizer (ViNormT5)
        self.normalizer = TextNormalizer(
            model_name=self.config.norm_model_name,
            max_length=128,
        )

        # text encoders (Phobert base)
        self.text_encoder = HFTextEncoder(
            model_name=self.config.text_model_name,
            max_length=self.config.max_desc_len,
        )
        self.comments_encoder = HFTextEncoder(
            model_name=self.config.text_model_name,
            max_length=self.config.max_comments_len,
        )

        # classifiers
        self.toxicity_clf = BinaryCommentClassifier(
            model_name=self.config.toxicity_model_name,
            max_length=256,
        )
        self.construct_clf = BinaryCommentClassifier(
            model_name=self.config.construct_model_name,
            max_length=256,
        )

        self.numeric_feature_names = [
            "likes_log",
            "comments_log",
            "shares_log",
            "bookmarks_log",
            "views_log",
            "like_rate",
            "comment_rate",
            "share_rate",
            "bookmark_rate",
            "engagement_rate",
            "has_views",
            "has_bookmarks",
            "age_days",
            "month_sin",
            "month_cos",
            "has_date",
            "n_comments",
            "avg_len_chars",
            "avg_len_words",
            "frac_has_laugh_emoji",
            "frac_has_question",
            "n_toxic_comments",
            "frac_toxic_comments",
            "n_constructive_comments",
            "frac_constructive_comments",
        ]
        self.scaler = StandardScaler()

    # ---------- text normalization pipeline ----------

    def _normalize_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.strip()
        if not text:
            return ""
        text = self.normalizer.normalize(text)
        text = map_emojis_to_tokens(text)
        text = normalize_whitespace(text)
        return text

    # ---------- numeric vector ----------

    def _build_numeric_vector(self, meta: Dict[str, Any]) -> np.ndarray:
        num_f = compute_numeric_features(meta)
        date_f = compute_date_features(
            meta.get("date"),
            self.config.reference_date,
            self.config.assume_year,
        )

        comments_tree = meta.get("comments_tree", []) or []
        flat_comments_raw = flatten_comments_tree(comments_tree)

        flat_comments = [self._normalize_text(c) for c in flat_comments_raw]
        comment_basic_f = compute_comment_basic_features(flat_comments)

        n_toxic = 0
        n_construct = 0
        for c in flat_comments:
            if not c:
                continue
            if self.toxicity_clf.predict_label(c) == 1:
                n_toxic += 1
            if self.construct_clf.predict_label(c) == 1:
                n_construct += 1

        n_all = len([c for c in flat_comments if c.strip() != ""])
        frac_toxic = float(n_toxic / n_all) if n_all > 0 else 0.0
        frac_construct = float(n_construct / n_all) if n_all > 0 else 0.0

        toxicity_f = {
            "n_toxic_comments": float(n_toxic),
            "frac_toxic_comments": frac_toxic,
            "n_constructive_comments": float(n_construct),
            "frac_constructive_comments": frac_construct,
        }

        merged: Dict[str, float] = {}
        merged.update(num_f)
        merged.update(date_f)
        merged.update(comment_basic_f)
        merged.update(toxicity_f)

        return np.array(
            [merged[name] for name in self.numeric_feature_names],
            dtype=np.float32,
        )

    # ---------- scaler ----------

    from tqdm import tqdm
    def fit_numeric_scaler(self, metas: List[Dict[str, Any]]) -> None:
        # Optimization: Fit on subset (max 50) to save time
        if len(metas) > 50:
             import random
             # Use fixed seed for reproducibility
             rs = random.Random(42)
             metas_sample = rs.sample(metas, 50)
        else:
             metas_sample = metas

        logger = logging.getLogger("preprocess-full")
        logger.info(f"Fitting numeric scaler on {len(metas_sample)} samples (subset of {len(metas)})...")
        rows = []
        for m in tqdm(metas_sample, desc="Fitting Scaler"):
            rows.append(self._build_numeric_vector(m))
        X = np.stack(rows, axis=0)
        self.scaler.fit(X)

    # ---------- text encodings ----------

    def _encode_description(self, meta: Dict[str, Any]) -> np.ndarray:
        raw_desc = meta.get("description", "") or ""
        clean_desc = strip_hashtags(raw_desc)
        clean_desc = self._normalize_text(clean_desc)
        return self.text_encoder.encode(clean_desc)

    def _encode_tags(self, meta: Dict[str, Any]) -> np.ndarray:
        tags = meta.get("tags", []) or []
        tag_tokens = [t.lstrip("#") for t in tags]
        tag_text = " ".join(tag_tokens)
        tag_text = self._normalize_text(tag_text)
        return self.text_encoder.encode(tag_text)

    def _encode_comments(self, meta: Dict[str, Any]) -> np.ndarray:
        comments_tree = meta.get("comments_tree", []) or []
        flat_comments_raw = flatten_comments_tree(comments_tree)
        if not flat_comments_raw:
            hidden_size = self.comments_encoder.model.config.hidden_size
            return np.zeros((hidden_size,), dtype=np.float32)

        flat_comments = [self._normalize_text(c) for c in flat_comments_raw]
        joined = " [SEP] ".join(flat_comments)
        return self.comments_encoder.encode(joined)

    # ---------- public API ----------

    def transform_single(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        numeric_raw = self._build_numeric_vector(meta)
        numeric_scaled = self.scaler.transform(numeric_raw[None, :])[0]

        desc_emb = self._encode_description(meta)
        tags_emb = self._encode_tags(meta)
        comments_emb = self._encode_comments(meta)

        return {
            "numeric_scaled": numeric_scaled,
            "numeric_raw": numeric_raw,
            "desc_emb": desc_emb,
            "tags_emb": tags_emb,
            "comments_emb": comments_emb,
        }
