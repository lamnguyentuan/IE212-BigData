from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from .video_utils import extract_frames_ffmpeg


@dataclass
class VideoFrameExtractor:
    """
    Extract frames từ Medallion Layout:
      bronze/{video_id}/video.mp4
      → silver/{video_id}/frames/frame_0001.jpg ...
    """
    base_dir: Path = Path("tiktok-data")
    bronze_name: str = "bronze"
    silver_name: str = "silver"

    num_frames: int = 16
    frame_size: tuple[int, int] = (224, 224)

    def bronze_dir(self) -> Path:
        return self.base_dir / self.bronze_name

    def silver_dir(self) -> Path:
        return self.base_dir / self.silver_name

    def list_video_ids(self) -> List[str]:
        bdir = self.bronze_dir()
        if not bdir.exists():
            return []
        return [
            p.name for p in bdir.iterdir()
            if p.is_dir() and (p / "video.mp4").exists()
        ]

    def extract_for_video_id(self, video_id: str):
        bronze_video_dir = self.bronze_dir() / video_id
        silver_video_dir = self.silver_dir() / video_id
        frames_dir = silver_video_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        video_path = bronze_video_dir / "video.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"video.mp4 missing for video_id={video_id}")

        extract_frames_ffmpeg(
            video_path=video_path,
            out_dir=frames_dir,
            frame_size=self.frame_size,
            num_frames=self.num_frames,
        )

    def extract_frames(self, video_path: Path, out_dir: Path, video_id: str = "") -> None:
        """
        Wrapper cho pipeline: chỉ định path cụ thể.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        extract_frames_ffmpeg(
            video_path=video_path,
            out_dir=out_dir,
            frame_size=self.frame_size,
            num_frames=self.num_frames,
        )

    def extract_all(self, video_ids: Iterable[str] | None = None):
        if video_ids is None:
            video_ids = self.list_video_ids()

        for vid in video_ids:
            print(f"[VideoFrameExtractor] Extracting frames for {vid}")
            try:
                self.extract_for_video_id(vid)
            except Exception as e:
                print(f"ERROR: {vid} — {e}")
