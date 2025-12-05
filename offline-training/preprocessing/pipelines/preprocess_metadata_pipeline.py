from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import json

import numpy as np
import yaml

from ..metadata import MetadataPreprocessor, load_metadata_config
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


def _load_all_metadata(bronze_dir: Path) -> List[Dict[str, Any]]:
    metas: List[Dict[str, Any]] = []
    if not bronze_dir.exists():
        return metas

    for meta_path in bronze_dir.glob("*/metadata.json"):
        try:
            video_id = meta_path.parent.name
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            raw["video_id"] = video_id
            metas.append(raw)
        except Exception as e:
            print(f"[MetadataPipeline] ERROR reading {meta_path}: {e}")
    return metas


def run_preprocess_metadata():
    logger = get_logger("preprocess-metadata")

    with Timer("preprocess_metadata_pipeline", logger_name="preprocess-metadata"):
        root = Path(__file__).resolve().parent.parent
        config_dir = root / "configs"

        paths_cfg = _load_yaml(config_dir / "paths.yaml")
        mp_cfg_path = config_dir / "metadata_preprocess.yaml"

        base_dir = Path(paths_cfg.get("data_root", "tiktok-data"))
        bronze_name = paths_cfg.get("bronze_subdir", "bronze")
        silver_name = paths_cfg.get("silver_subdir", "silver")

        use_minio = bool(paths_cfg.get("use_minio", False))
        upload_silver = bool(paths_cfg.get("upload_silver", False))
        minio_bucket = paths_cfg.get("minio_bucket", None)

        bronze_dir = base_dir / bronze_name
        silver_dir = base_dir / silver_name

        logger.info(
            f"base_dir={base_dir}, bronze={bronze_name}, silver={silver_name}, "
            f"use_minio={use_minio}, upload_silver={upload_silver}"
        )

        # -------- MinIO sync: download bronze nếu bật --------
        if use_minio:
            cfg = MinioConfig.from_env(bucket=minio_bucket)
            mc = MinioClientWrapper(cfg)

            # Nếu paths.yaml có video_ids thì sync từng cái,
            # còn không thì thôi, assume đã sync từ audio/video pipeline.
            ids_cfg = paths_cfg.get("video_ids")
            if isinstance(ids_cfg, list) and ids_cfg:
                logger.info(f"[MinIO] Syncing bronze for {len(ids_cfg)} videos...")
                for vid in ids_cfg:
                    try:
                        mc.download_bronze_video(str(vid), base_dir)
                    except Exception as e:
                        logger.error(f"[MinIO] Failed to download bronze for {vid}: {e}")

        # -------- Load metadata từ bronze --------
        cfg = load_metadata_config(mp_cfg_path)
        pre = MetadataPreprocessor(config=cfg)

        metas = _load_all_metadata(bronze_dir)
        if not metas:
            logger.warning("[MetadataPipeline] No metadata found.")
            return

        logger.info(f"Fitting scaler on {len(metas)} samples...")
        pre.fit_numeric_scaler(metas)

        processed_ids: List[str] = []

        for meta in metas:
            vid = meta["video_id"]
            try:
                logger.info(f"Processing metadata for video_id={vid}")
                out = pre.transform_single(meta)
                vdir = silver_dir / vid
                vdir.mkdir(parents=True, exist_ok=True)

                npz_path = vdir / "metadata_features.npz"
                np.savez_compressed(
                    npz_path,
                    numeric_scaled=out["numeric_scaled"],
                    numeric_raw=out["numeric_raw"],
                    desc_emb=out["desc_emb"],
                    tags_emb=out["tags_emb"],
                    comments_emb=out["comments_emb"],
                )
                logger.info(f"Saved metadata_features for {vid} to {npz_path}")
                processed_ids.append(vid)
            except Exception as e:
                logger.error(f"[MetadataPipeline] ERROR for {vid}: {e}")

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
    run_preprocess_metadata()
