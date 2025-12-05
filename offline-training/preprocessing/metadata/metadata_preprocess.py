# metadata_preprocess.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import re
import math
import datetime as dt

import numpy as np
from sklearn.preprocessing import StandardScaler

import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)

########################################
# 1. Emoji mapping & text utilities    #
########################################

LAUGH_EMOJIS = {"😂", "🤣", "😄", "😆", "😅", "😹", "😸"}
SAD_EMOJIS = {"😔", "😢", "😭", "😞", "☹️", "🙁"}
SCARE_EMOJIS = {"😱", "🤯", "😨", "😰", "😧"}


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


########################################
# 2. Numeric parsing & ratio features  #
########################################

def parse_count(value: Optional[str]) -> Optional[float]:
    """Parse các trường likes/shares/... dạng '307K', '4.5M', '3537', 'N/A'."""
    if value is None:
        return None
    value = value.strip()
    if value == "" or value.upper() == "N/A":
        return None

    v = value.upper()
    try:
        if v.endswith("K"):
            return float(v[:-1]) * 1e3
        if v.endswith("M"):
            return float(v[:-1]) * 1e6
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

    # log features
    likes_log = safe_log1p(likes)
    comments_log = safe_log1p(comments)
    shares_log = safe_log1p(shares)
    bookmarks_log = safe_log1p(bookmarks)
    views_log = safe_log1p(views)

    # ratios
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


########################################
# 3. Date handling
########################################

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


########################################
# 4. ViNormT5 – Text Normalizer
########################################

@dataclass
class TextNormalizer:
    model_name: str = "meoo225/ViNormT5"
    max_length: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def normalize(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        # prefix giống code gốc
        input_text = "text-normalize: " + text
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                inputs,
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True,
            )
        output_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )
        return output_text


########################################
# 5. HFTextEncoder – Phobert Embedding
########################################

@dataclass
class HFTextEncoder:
    model_name: str = "vinai/phobert-base-v2"
    max_length: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, text: str) -> np.ndarray:
        """Encode 1 chuỗi text → embedding (mean pooling last hidden state)."""
        text = text or ""
        text = text.strip()
        if not text:
            hidden_size = self.model.config.hidden_size
            return np.zeros((hidden_size,), dtype=np.float32)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state  # (1, L, H)
            mask = inputs["attention_mask"].unsqueeze(-1)  # (1, L, 1)
            masked = last_hidden * mask
            summed = masked.sum(dim=1)  # (1, H)
            counts = mask.sum(dim=1)  # (1, 1)
            counts = torch.clamp(counts, min=1e-5)
            pooled = (summed / counts).squeeze(0)  # (H,)

        return pooled.cpu().numpy().astype(np.float32)


########################################
# 6. Binary classifier cho Toxic / Constructive
########################################

@dataclass
class BinaryCommentClassifier:
    model_name: str
    max_length: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        self.model.to(self.device)
        self.model.eval()

    def predict_label(self, text: str) -> int:
        """
        Trả về 0 hoặc 1 (giống label_map trong code của bạn).
        Nếu text rỗng → 0.
        """
        if not isinstance(text, str) or text.strip() == "":
            return 0

        # giữ logic giới hạn độ dài ~600 nếu muốn
        if len(text) > 600:
            text = text[:600]

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()

        # label_map {0: 0.0, 1: 1.0} ⇒ trả về int(0/1)
        return int(predicted_class)


########################################
# 7. Comments flatten & basic features
########################################

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


########################################
# 8. MetadataPreprocessor – dùng 3 model bạn đưa
########################################

@dataclass
class MetadataPreprocessor:
    # text embedding
    text_model_name: str = "vinai/phobert-base-v2"
    # toxicity / constructiveness
    toxicity_model_name: str = "funa21/phobert-finetuned-victsd-toxic-v2"
    construct_model_name: str = "funa21/phobert-finetuned-victsd-constructiveness-v2"
    # normalizer
    norm_model_name: str = "meoo225/ViNormT5"

    # date config
    reference_date: dt.date = dt.date(2025, 12, 5)
    assume_year: int = 2025

    max_desc_len: int = 128
    max_tags_len: int = 64
    max_comments_len: int = 256

    def __post_init__(self):
        # text normalizer (ViNormT5)
        self.normalizer = TextNormalizer(
            model_name=self.norm_model_name,
            max_length=128,
        )

        # text encoders (Phobert base)
        self.text_encoder = HFTextEncoder(
            model_name=self.text_model_name,
            max_length=self.max_desc_len,
        )
        self.comments_encoder = HFTextEncoder(
            model_name=self.text_model_name,
            max_length=self.max_comments_len,
        )

        # classifiers
        self.toxicity_clf = BinaryCommentClassifier(
            model_name=self.toxicity_model_name,
            max_length=256,
        )
        self.construct_clf = BinaryCommentClassifier(
            model_name=self.construct_model_name,
            max_length=256,
        )

        # list feature numeric
        self.numeric_feature_names = [
            # numeric + ratio
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
            # date
            "age_days",
            "month_sin",
            "month_cos",
            "has_date",
            # comments basic
            "n_comments",
            "avg_len_chars",
            "avg_len_words",
            "frac_has_laugh_emoji",
            "frac_has_question",
            # toxicity / constructive
            "n_toxic_comments",
            "frac_toxic_comments",
            "n_constructive_comments",
            "frac_constructive_comments",
        ]
        self.scaler = StandardScaler()

    # ---------- text normalization pipeline ----------

    def _normalize_text(self, text: str) -> str:
        """Chuẩn hóa bằng ViNormT5 + mapping emoji + clean whitespace."""
        if not isinstance(text, str):
            return ""
        text = text.strip()
        if not text:
            return ""
        # normalize bằng ViNormT5
        text = self.normalizer.normalize(text)
        # map emoji → token
        text = map_emojis_to_tokens(text)
        # clean space
        text = normalize_whitespace(text)
        return text

    # ---------- numeric vector (log, ratio, comment stats, tox/construct) ----------

    def _build_numeric_vector(self, meta: Dict[str, Any]) -> np.ndarray:
        num_f = compute_numeric_features(meta)
        date_f = compute_date_features(
            meta.get("date"), self.reference_date, self.assume_year
        )

        comments_tree = meta.get("comments_tree", []) or []
        flat_comments_raw = flatten_comments_tree(comments_tree)

        # chuẩn hóa từng comment bằng ViNormT5
        flat_comments = [self._normalize_text(c) for c in flat_comments_raw]

        comment_basic_f = compute_comment_basic_features(flat_comments)

        # toxicity & constructive cho từng comment
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

    # ---------- fit scaler trên toàn dataset ----------

    def fit_numeric_scaler(self, metas: List[Dict[str, Any]]) -> None:
        rows = []
        for m in metas:
            rows.append(self._build_numeric_vector(m))
        X = np.stack(rows, axis=0)
        self.scaler.fit(X)

    # ---------- text embeddings ----------

    def _encode_description(self, meta: Dict[str, Any]) -> np.ndarray:
        raw_desc = meta.get("description", "") or ""
        # bỏ hashtag trong desc
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
        """
        Tiền xử lý 1 metadata JSON -> dict:
          - numeric_scaled: np.ndarray (F_num,)
          - numeric_raw:    np.ndarray (F_num,)
          - desc_emb:       np.ndarray (H,)
          - tags_emb:       np.ndarray (H,)
          - comments_emb:   np.ndarray (H,)
        """
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


########################################
# 9. Demo nhỏ
########################################

if __name__ == "__main__":
    import json

    raw_json = r'''
    {
      "likes": "307K",
      "comments": "3537",
      "shares": "493K",
      "bookmarks": "7389",
      "views": "N/A",
      "description": "😄😄😄😄😄😄#funny  #funnyvideos  #fyp  #xh  #haihuoc  #vuinhon",
      "musicTitle": "N/A",
      "date": "8-9",
      "author": "N/A",
      "tags": [
        "#funny",
        "#funnyvideos",
        "#fyp",
        "#xh",
        "#haihuoc",
        "#vuinhon"
      ],
      "videoUrl": "https://www.tiktok.com/@hohoppe/video/7536570873982061837",
      "comments_tree": [
        {
          "text": "Sao mày ngồi đây vậy Tâm",
          "replies": [
            {"text": "@Mỹ Tâm"},
            {"text": "@Kiều Tâm🌻"},
            {"text": "Anh nhắc em🤓"}
          ]
        }
      ],
      "video_s3_path": "s3://tiktok-data/bronze/7536570873982061837/video.mp4"
    }
    '''

    meta = json.loads(raw_json)

    pre = MetadataPreprocessor()
    # Thực tế: fit_numeric_scaler trên toàn bộ dataset trước
    pre.fit_numeric_scaler([meta])

    out = pre.transform_single(meta)
    print("numeric_scaled shape:", out["numeric_scaled"].shape)
    print("desc_emb shape:", out["desc_emb"].shape)
    print("tags_emb shape:", out["tags_emb"].shape)
    print("comments_emb shape:", out["comments_emb"].shape)
