from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import subprocess


def run_cmd(cmd: list[str], timeout: float = 300.0):
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False
        )
    except subprocess.TimeoutExpired as e:
         raise RuntimeError(
            f"Command timed out after {timeout}s: {' '.join(cmd)}\n"
            f"stderr: {e.stderr.decode('utf-8', errors='ignore') if e.stderr else ''}"
        ) from e

    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stderr: {proc.stderr.decode('utf-8', errors='ignore')}"
        )


def extract_frames_ffmpeg(
    video_path: Path,
    out_dir: Path,
    frame_size: tuple[int, int] = (224, 224),
    num_frames: int = 16,
):
    """
    Dùng ffmpeg lấy EXACT số lượng frames.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = frame_size

    # Lệnh ffmpeg:
    # -vf "scale=WIDTH:HEIGHT, fps=NUM_FRAMES/VIDEO_DURATION" là nguy hiểm.
    # => Cách tốt nhất: trích ALL frames rồi tự sample.
    temp_dir = out_dir / "_all_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract tất cả frames
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel", "error",
        "-i",
        str(video_path),
        "-vf",
        f"scale={w}:{h}",
        str(temp_dir / "frame_%05d.jpg"),
        "-y",
    ]
    run_cmd(cmd, timeout=300)

    # 2. Uniform sampling
    all_frames = sorted(temp_dir.glob("frame_*.jpg"))
    if len(all_frames) == 0:
        raise RuntimeError(f"No frames extracted from {video_path}")

    indices = np.linspace(0, len(all_frames) - 1, num_frames).astype(int)
    sampled = [all_frames[idx] for idx in indices]

    # Copy sampled frames ra thư mục output
    for i, frame_path in enumerate(sampled):
        target = out_dir / f"frame_{i+1:04d}.jpg"
        target.write_bytes(frame_path.read_bytes())

    # cleanup
    for f in all_frames:
        f.unlink()
    temp_dir.rmdir()


def load_image(path: Path) -> np.ndarray:
    """Load ảnh dưới dạng RGB."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to read image {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
