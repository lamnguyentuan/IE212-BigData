from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from ..audio import AudioExtractor, Wav2Vec2AudioEncoder
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
    """
    Nếu paths.yaml có:
        video_ids: ["id1", "id2", ...]
    thì dùng list đó.

    Nếu không, tự scan local:
        bronze/{video_id}/video.mp4
    """
    ids_cfg = paths_cfg.get("video_ids")
    if isinstance(ids_cfg, list) and ids_cfg:
        return [str(v) for v in ids_cfg]

    # fallback: scan local bronze
    video_ids: List[str] = []
    if bronze_dir.exists():
        for p in bronze_dir.iterdir():
            if p.is_dir() and (p / "video.mp4").exists():
                video_ids.append(p.name)
    return sorted(video_ids)


def run_preprocess_audio():
    logger = get_logger("preprocess-audio")

    with Timer("preprocess_audio_pipeline", logger_name="preprocess-audio"):
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

        sample_rate = int(pre_cfg.get("audio_sample_rate", 16000))
        audio_model_name = (
            (enc_cfg.get("audio", {}) or {}).get("model_name", "facebook/wav2vec2-base")
        )

        logger.info(
            f"base_dir={base_dir}, bronze={bronze_name}, silver={silver_name}, "
            f"use_minio={use_minio}, upload_silver={upload_silver}"
        )
        logger.info(f"audio_sample_rate={sample_rate}, audio_model={audio_model_name}")

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

        # -------- Step 1: Extract audio.wav -> silver --------
        extractor = AudioExtractor(
            base_dir=base_dir,
            bronze_name=bronze_name,
            silver_name=silver_name,
            sample_rate=sample_rate,
        )
        if video_ids:
            extractor.extract_all(video_ids)
        else:
            extractor.extract_all()

        # -------- Step 2: Encode audio_embedding.npy --------
        encoder = Wav2Vec2AudioEncoder(model_name=audio_model_name)
        embeddings = encoder.encode_silver_folder(silver_dir)

        for vid, emb in embeddings.items():
            out_path = silver_dir / vid / "audio_embedding.npy"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, emb)
            logger.info(f"Saved audio_embedding for {vid} to {out_path}")

        # -------- MinIO sync: upload silver nếu bật --------
        if use_minio and upload_silver:
            logger.info(f"[MinIO] Uploading silver/* for {len(embeddings)} videos...")
            for vid in embeddings.keys():
                try:
                    mc.upload_silver_dir(vid, base_dir)
                except Exception as e:
                    logger.error(f"[MinIO] Failed to upload silver for {vid}: {e}")


if __name__ == "__main__":
    run_preprocess_audio()
