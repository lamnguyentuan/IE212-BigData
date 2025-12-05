from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from .video_utils import load_image


@dataclass
class VideoFrameLoader:
    """
    Load frames → tensor shape (T, C, H, W).
    Normalize theo ImageNet mean/std.
    """
    frame_size: tuple[int, int] = (224, 224)
    num_frames: int = 16

    # ImageNet mean/std
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def load_frames(self, frames_dir: Path) -> torch.Tensor:
        all_frames = sorted(frames_dir.glob("frame_*.jpg"))
        if len(all_frames) < self.num_frames:
            raise RuntimeError(f"Expected {self.num_frames} frames, got {len(all_frames)}")

        frames = all_frames[:self.num_frames]
        imgs = [load_image(f) for f in frames]

        # Convert to numpy: (T, H, W, C)
        imgs = np.stack(imgs, axis=0).astype(np.float32) / 255.0

        # Normalize
        imgs = (imgs - np.array(self.mean)) / np.array(self.std)

        # To tensor → (T, C, H, W)
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2)
        return imgs
