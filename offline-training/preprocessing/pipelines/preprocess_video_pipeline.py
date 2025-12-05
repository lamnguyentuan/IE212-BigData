from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from ..video import VideoFrameExtractor, VideoFrameLoader, TimeSformerVideoEncoder
from ..utils import (
    get_logger,
    Timer,
    MinioConfig,
    MinioClientWrapper,
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_video_ids_from_paths_cfg(
    paths_cfg: Dict[str, Any],
    bronze_dir: Path,
) -> List[str]:
    ids_cfg = paths_cfg.get("video_ids")
    if isinstance(ids_cfg, list) and ids_cfg:
        return [str(v) for v in ids_cfg]

    video_ids: List[str] = []
    if bronze_dir.exists():
        for p in bronze_dir.iterdir():
            if p.is_dir() and (p / "video.mp4").exists():
                video_ids.append(p.name)
    return sorted(video_ids)


def run_preprocess_video():
    logger = get_logger("preprocess-video")

    with Timer("preprocess_video_pipeline", logger_name="preprocess-video"):
        root = Path(__file__).resolve().parent.parent
        config_dir = root / "configs"

        paths_cfg = _load_yaml(config_dir / "paths.yaml")
        pre_cfg = _load_yaml(config_dir / "preprocess_config.yaml")
        enc_cfg = _load_yaml(config_dir / "encoders.yaml")

        base_dir = Path(paths_cfg.get("data_root", "tiktok-data"))
        bronze_name = paths_cfg.get("bronze_subdir", "bronze")
        silver_name = paths_cfg.get("silver_subdir", "silver")

        use_minio = bool(paths_cfg.get("use_minio", False))
        upload_silver = bool(paths_cfg.get("upload_silver", False))
        minio_bucket = paths_cfg.get("minio_bucket", None)

        bronze_dir = base_dir / bronze_name
        silver_dir = base_dir / silver_name

        num_frames = int(pre_cfg.get("num_frames", 16))
        frame_size = tuple(pre_cfg.get("frame_size", [224, 224]))
        video_model_name = (
            (enc_cfg.get("video", {}) or {}).get(
                "model_name", "facebook/timesformer-base-finetuned-k400"
            )
        )

        logger.info(
            f"base_dir={base_dir}, bronze={bronze_name}, silver={silver_name}, "
            f"use_minio={use_minio}, upload_silver={upload_silver}"
        )
        logger.info(
            f"num_frames={num_frames}, frame_size={frame_size}, video_model={video_model_name}"
        )

        # -------- MinIO sync: download bronze nếu bật --------
        video_ids = _load_video_ids_from_paths_cfg(paths_cfg, bronze_dir)
        if use_minio:
            cfg = MinioConfig.from_env(bucket=minio_bucket)
            mc = MinioClientWrapper(cfg)

            logger.info(f"[MinIO] Syncing bronze for {len(video_ids)} videos...")
            for vid in video_ids:
                try:
                    mc.download_bronze_video(vid, base_dir)
                except Exception as e:
                    logger.error(f"[MinIO] Failed to download bronze for {vid}: {e}")
        else:
            mc = None  # chỉ để dùng sau nếu upload_silver=True (nhưng use_minio=False thì bỏ qua)

        # -------- Step 1: Extract frames --------
        extractor = VideoFrameExtractor(
            base_dir=base_dir,
            bronze_name=bronze_name,
            silver_name=silver_name,
            num_frames=num_frames,
            frame_size=frame_size,
        )
        if video_ids:
            extractor.extract_all(video_ids)
        else:
            extractor.extract_all()

        # -------- Step 2: Encode video_embedding.npy --------
        loader = VideoFrameLoader(num_frames=num_frames, frame_size=frame_size)
        encoder = TimeSformerVideoEncoder(model_name=video_model_name)

        processed_ids: List[str] = []

        for video_dir in silver_dir.iterdir():
            if not video_dir.is_dir():
                continue
            frames_dir = video_dir / "frames"
            if not frames_dir.exists():
                continue
            vid = video_dir.name
            try:
                logger.info(f"Encoding video_id={vid}")
                frames = loader.load_frames(frames_dir)
                emb = encoder.encode(frames)
                out_path = video_dir / "video_embedding.npy"
                np.save(out_path, emb)
                logger.info(f"Saved video_embedding to {out_path}")
                processed_ids.append(vid)
            except Exception as e:
                logger.error(f"ERROR for video_id={vid}: {e}")

        # -------- MinIO sync: upload silver nếu bật --------
        if use_minio and upload_silver and processed_ids:
            cfg = MinioConfig.from_env(bucket=minio_bucket)
            mc = MinioClientWrapper(cfg)
            logger.info(f"[MinIO] Uploading silver/* for {len(processed_ids)} videos...")
            for vid in processed_ids:
                try:
                    mc.upload_silver_dir(vid, base_dir)
                except Exception as e:
                    logger.error(f"[MinIO] Failed to upload silver for {vid}: {e}")


if __name__ == "__main__":
    run_preprocess_video()
