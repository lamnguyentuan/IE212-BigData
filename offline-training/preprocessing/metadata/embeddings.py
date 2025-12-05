from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

from .text_utils import map_emojis_to_tokens, normalize_whitespace


@dataclass
class TextNormalizer:
    model_name: str = "meoo225/ViNormT5"
    max_length: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def normalize(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
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
        return normalize_whitespace(map_emojis_to_tokens(output_text))


@dataclass
class HFTextEncoder:
    model_name: str = "vinai/phobert-base-v2"
    max_length: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, text: str) -> np.ndarray:
        """Encode 1 chuỗi text → embedding (mean pooling last hidden state)."""
        text = (text or "").strip()
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
