from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Dict, Any, List

import numpy as np
import torch
from transformers import AutoProcessor, AutoModel

from .audio_utils import load_wav_mono_16k


@dataclass
class Wav2Vec2AudioEncoder:
    """
    Encoder audio dùng HuggingFace Wav2Vec2.

    - Mặc định dùng AutoProcessor + AutoModel để tạo embedding.
    - Pooling: mean pooling theo time dimension (last_hidden_state).
    """
    model_name: str = "facebook/wav2vec2-base"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self.hidden_size = self.model.config.hidden_size

    def _encode_waveform(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """
        Encode 1 waveform (np.ndarray shape (T,)) → embedding (H,).
        """
        # processor có thể tự resample nếu cần, nhưng tốt nhất là 16k
        inputs = self.processor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state  # (B, T, H)
            # mean pooling theo time
            emb = hidden.mean(dim=1)           # (B, H)
            emb = emb[0]                       # (H,)

        return emb.cpu().numpy().astype(np.float32)

    def encode_file(self, wav_path: Path) -> np.ndarray:
        """
        Encode 1 file wav → embedding vector (H,).
        """
        waveform, sr = load_wav_mono_16k(wav_path)
        return self._encode_waveform(waveform, sr)

    def encode_files(self, wav_paths: Iterable[Path]) -> Dict[str, np.ndarray]:
        """
        Encode nhiều file wav, return dict:
            { "video_id": embedding }
        Giả định wav file nằm trong silver/{video_id}/audio.wav.
        """
        results: Dict[str, np.ndarray] = {}
        for wp in wav_paths:
            video_id = wp.parent.name
            print(f"[Wav2Vec2AudioEncoder] Encoding {video_id} from {wp}")
            emb = self.encode_file(wp)
            results[video_id] = emb
        return results

    def encode_silver_folder(self, silver_dir: Path) -> Dict[str, np.ndarray]:
        """
        Scan qua silver/*/audio.wav và encode tất cả.
        """
        wav_paths: List[Path] = []
        for video_dir in silver_dir.iterdir():
            if not video_dir.is_dir():
                continue
            wav_path = video_dir / "audio.wav"
            if wav_path.exists():
                wav_paths.append(wav_path)

        return self.encode_files(wav_paths)
