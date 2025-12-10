from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from .audio_utils import extract_audio_ffmpeg


@dataclass
class AudioExtractor:
    """
    Extractor đơn giản:
      - input: cây thư mục Medallion:
            base_dir/
              ├── bronze/{video_id}/video.mp4
              └── silver/{video_id}/ (sẽ được tạo nếu chưa có)
      - output: silver/{video_id}/audio.wav
    """
    base_dir: Path = Path("tiktok-data")
    bronze_name: str = "bronze"
    silver_name: str = "silver"
    sample_rate: int = 16000

    def bronze_dir(self) -> Path:
        return self.base_dir / self.bronze_name

    def silver_dir(self) -> Path:
        return self.base_dir / self.silver_name

    def list_video_ids(self) -> List[str]:
        """Liệt kê tất cả video_id có trong bronze/."""
        bdir = self.bronze_dir()
        if not bdir.exists():
            return []
        return sorted(
            p.name for p in bdir.iterdir()
            if p.is_dir() and (p / "video.mp4").exists()
        )

    def extract_for_video_id(self, video_id: str) -> Path:
        """
        Tách audio cho 1 video_id.
        Return: đường dẫn file wav output.
        """
        bronze_video_dir = self.bronze_dir() / video_id
        silver_video_dir = self.silver_dir() / video_id
        silver_video_dir.mkdir(parents=True, exist_ok=True)

        video_path = bronze_video_dir / "video.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"video.mp4 not found for video_id={video_id}")

        out_wav = silver_video_dir / "audio.wav"
        extract_audio_ffmpeg(
            video_path=video_path,
            out_wav_path=out_wav,
            sample_rate=self.sample_rate,
        )
        return out_wav

    def extract_audio_for_video(self, video_path: Path, out_dir: Path, video_id: str = "") -> Path:
        """
        Wrapper cho pipeline: cho phép chỉ định path cụ thể mà không infer theo base_dir.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        out_wav = out_dir / "audio.wav"
        extract_audio_ffmpeg(
            video_path=video_path,
            out_wav_path=out_wav,
            sample_rate=self.sample_rate,
        )
        return out_wav

    def extract_all(self, video_ids: Iterable[str] | None = None) -> None:
        """
        Chạy extract audio cho nhiều video.
        Nếu video_ids=None → tự scan bronze/*/.
        """
        if video_ids is None:
            video_ids = self.list_video_ids()

        for vid in video_ids:
            print(f"[AudioExtractor] Extracting audio for video_id={vid}")
            try:
                self.extract_for_video_id(vid)
            except Exception as e:
                print(f"[AudioExtractor] ERROR for {vid}: {e}")
