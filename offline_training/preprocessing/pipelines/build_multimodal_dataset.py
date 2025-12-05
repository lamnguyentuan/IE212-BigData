from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import yaml

from ..features.multimodal_feature_builder import MultimodalFeatureBuilder
from ..features.feature_saver import FeatureSaver


logger = logging.getLogger("build-multimodal")
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s - %(message)s",
)


def _load_paths_config() -> dict:
    """
    Đọc file configs/paths.yaml (tương đối so với thư mục preprocessing).
    """
    preprocessing_dir = Path(__file__).resolve().parents[1]
    config_path = preprocessing_dir / "configs" / "paths.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"paths.yaml not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _discover_video_ids(silver_dir: Path) -> List[str]:
    """
    Nếu paths.yaml không chỉ rõ video_ids, tự scan trong silver/*.
    """
    if not silver_dir.exists():
        logger.warning(f"silver dir not found: {silver_dir}")
        return []
    ids = []
    for p in silver_dir.iterdir():
        if p.is_dir():
            # kiểm tra có đủ file cơ bản chưa
            has_video = (p / "video_embedding.npy").exists()
            has_audio = (p / "audio_embedding.npy").exists()
            has_meta = (p / "metadata_features.npz").exists()
            if has_video and has_audio and has_meta:
                ids.append(p.name)
    return sorted(ids)


def run_build_multimodal_dataset() -> Path:
    """
    Entry chính được gọi khi chạy:
        python -m offline_training.preprocessing.pipelines.build_multimodal_dataset
    """
    paths_cfg = _load_paths_config()

    base_dir = Path(paths_cfg.get("data_root", "tiktok-data"))
    silver_name = paths_cfg.get("silver_subdir", "silver")
    gold_name = paths_cfg.get("gold_subdir", "gold")
    bronze_name = paths_cfg.get("bronze_subdir", "bronze")
    use_minio = bool(paths_cfg.get("use_minio", False))

    logger.info(
        "build-multimodal - base_dir=%s, silver=%s, gold=%s, use_minio=%s",
        base_dir,
        silver_name,
        gold_name,
        use_minio,
    )
    if use_minio:
        logger.info(
            "build-multimodal - [MinIO] NOTE: build_multimodal_dataset currently assumes silver is already local."
        )

    silver_dir = base_dir / silver_name
    gold_dir = base_dir / gold_name
    gold_dir.mkdir(parents=True, exist_ok=True)

    # Lấy danh sách video_ids
    explicit_ids = paths_cfg.get("video_ids", [])
    if explicit_ids:
        video_ids = explicit_ids
    else:
        video_ids = _discover_video_ids(silver_dir)

    if not video_ids:
        raise ValueError(
            "No video_ids found. Either set video_ids in paths.yaml or ensure silver/* contains processed videos."
        )

    # Khởi tạo builder + saver
    builder = MultimodalFeatureBuilder(
        base_dir=base_dir,
        silver_name=silver_name,
        gold_name=gold_name,
        bronze_name=bronze_name,
    )
    saver = FeatureSaver(
        base_dir=base_dir,
        gold_name=gold_name,
        use_minio=use_minio,
        upload_gold=bool(paths_cfg.get("upload_gold", False)),
        minio_bucket=paths_cfg.get("minio_bucket"),
    )


    # Build rows
    rows = builder.build_rows(video_ids)
    logger.info("build-multimodal - Built %d multimodal rows.", len(rows))

    if not rows:
        raise ValueError("No rows to save in build_multimodal_dataset. Check that silver/* has all required files.")

    # Tên dataset: có thể tuỳ biến, ở đây lấy theo tên base_dir
    dataset_name = f"{base_dir.name}_multimodal"
    out_path = saver.save_npz(dataset_name, rows)
    logger.info("build-multimodal - Saved dataset to %s", out_path)

    # Nếu bạn có logic upload MinIO trong FeatureSaver, nó sẽ tự chạy.
    return out_path


if __name__ == "__main__":
    run_build_multimodal_dataset()
