from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf  # pip install soundfile


def run_cmd(cmd: list[str]) -> None:
    """Chạy 1 lệnh subprocess đơn giản, raise nếu lỗi."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): "
            f"{' '.join(cmd)}\n"
            f"stderr: {proc.stderr.decode('utf-8', errors='ignore')}"
        )


def extract_audio_ffmpeg(
    video_path: Path,
    out_wav_path: Path,
    sample_rate: int = 16000,
) -> None:
    """
    Tách audio từ video bằng ffmpeg, chuẩn hóa:
      - mono
      - sample_rate Hz (mặc định 16k)
    """
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vn",          # no video
        "-ac", "1",     # mono
        "-ar", str(sample_rate),
        "-f", "wav",
        str(out_wav_path),
        "-y",           # overwrite
    ]
    run_cmd(cmd)


def load_wav_mono_16k(
    wav_path: Path,
    expected_sr: int = 16000,
) -> tuple[np.ndarray, int]:
    """
    Đọc file wav thành:
      - waveform: np.ndarray shape (T,)
      - sample_rate

    Giả định audio đã được ffmpeg chuẩn hóa 16kHz mono, nên chỉ check lại.
    """
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    data, sr = sf.read(str(wav_path))
    # data có thể shape (T,) hoặc (T, C)
    if data.ndim == 2:
        # average channels -> mono
        data = data.mean(axis=1)

    if sr != expected_sr:
        # Không resample ở đây để giữ đơn giản, chỉ warning
        print(
            f"[WARN] Audio sample rate {sr} != expected {expected_sr}. "
            "Wav2Vec2 vẫn có thể chạy nhưng nên đảm bảo ffmpeg chuẩn hóa đúng."
        )

    # convert to float32
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    return data, sr
