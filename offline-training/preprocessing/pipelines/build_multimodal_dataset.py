from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

import yaml

from ..features import MultimodalFeatureBuilder, FeatureSaver
from ..utils import get_logger, Timer, MinioConfig, MinioClientWrapper


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def run_build_multimodal_dataset(dataset_name: str = "multimodal_dataset"):
    logger = get_logger("build-multimodal")

    with Timer("build_multimodal_dataset", logger_name="build-multimodal"):
        root = Path(__file__).resolve().parent.parent
        config_dir = root / "configs"

        paths_cfg = _load_yaml(config_dir / "paths.yaml")

        base_dir = Path(paths_cfg.get("data_root", "tiktok-data"))
        silver_name = paths_cfg.get("silver_subdir", "silver")
        gold_name = paths_cfg.get("gold_subdir", "gold")

        use_minio = bool(paths_cfg.get("use_minio", False))
        minio_bucket = paths_cfg.get("minio_bucket", None)

        logger.info(
            f"base_dir={base_dir}, silver={silver_name}, gold={gold_name}, "
            f"use_minio={use_minio}"
        )

        # 1) Nếu cần, sync silver từ MinIO về (optional, tuỳ bạn muốn logic nào)
        if use_minio:
            cfg = MinioConfig.from_env(bucket=minio_bucket)
            mc = MinioClientWrapper(cfg)
            # Ở đây mình không tự động sync all silver/* vì có thể rất nặng,
            # bạn có thể tự quyết định: hoặc bỏ qua, hoặc sync 1 list video theo paths.yaml.
            logger.info("[MinIO] NOTE: build_multimodal_dataset currently assumes silver is already local.")

        # 2) Build rows từ Silver
        builder = MultimodalFeatureBuilder(
            base_dir=base_dir,
            silver_name=silver_name,
            gold_name=gold_name,
        )
        rows = builder.build_rows()
        logger.info(f"Built {len(rows)} multimodal rows.")

        # 3) Save dataset vào gold/{dataset_name}.npz
        saver = FeatureSaver(
            base_dir=base_dir,
            gold_name=gold_name,
        )
        out_path = saver.save_npz(dataset_name, rows)

        # 4) Optionally upload file dataset lên MinIO
        if use_minio:
            cfg = MinioConfig.from_env(bucket=minio_bucket)
            mc = MinioClientWrapper(cfg)
            rel = out_path.relative_to(base_dir)
            object_name = str(rel).replace("\\", "/")
            mc.upload_file(out_path, object_name)


if __name__ == "__main__":
    run_build_multimodal_dataset()
