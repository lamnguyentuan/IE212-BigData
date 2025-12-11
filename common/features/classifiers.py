from __future__ import annotations
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class BinaryCommentClassifier:
    model_name: str
    max_length: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        self.model.to(self.device)
        self.model.eval()

    def predict_label(self, text: str) -> int:
        """
        Trả về 0 hoặc 1.
        Nếu text rỗng → 0.
        """
        if not isinstance(text, str) or text.strip() == "":
            return 0

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

        return int(predicted_class)
