from __future__ import annotations

import logging
import sys
from pathlib import Path
import os
from typing import List, Dict, Any
import json
import numpy as np
import yaml

# Thêm project root
FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from offline_training.preprocessing.audio.audio_extractor import AudioExtractor
from offline_training.preprocessing.audio.audio_encoder_wav2vec import Wav2Vec2AudioEncoder
from offline_training.preprocessing.video.video_frame_extractor import VideoFrameExtractor
from offline_training.preprocessing.video.video_loader import VideoFrameLoader
from offline_training.preprocessing.video.video_encoder_timesformer import TimeSformerVideoEncoder
from common.features.preprocessor import MetadataPreprocessor
from common.utils.minio_utils import MinioConfig, MinioClientWrapper
from common.utils.file_io import safe_rmtree

# Feature Builders
from common.features.multimodal_feature_builder import MultimodalFeatureBuilder
from offline_training.preprocessing.features.feature_saver import FeatureSaver
from common.features.feature_schema import MultimodalFeatureRow

logger = logging.getLogger("preprocess-full")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
)

def _load_paths_config() -> dict:
    preprocessing_dir = Path(__file__).resolve().parents[1]
    
    # Allow override via env var
    env_config = os.getenv("PREPROCESS_CONFIG_PATH")
    if env_config:
        config_path = ROOT / env_config
        if not config_path.exists():
             # Try relative to preprocessing dir
             config_path = preprocessing_dir / env_config
    else:
        config_path = preprocessing_dir / "configs" / "paths.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"paths.yaml not found at {config_path}")
    
    logger.info(f"Loading paths config from: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _init_minio(paths_cfg: dict) -> MinioClientWrapper | None:
    use_minio = bool(paths_cfg.get("use_minio", False))
    if not use_minio:
        return None
    bucket = paths_cfg.get("minio_bucket")
    if not bucket:
        raise ValueError("minio_bucket must be set when use_minio=True")
    cfg = MinioConfig.from_env(bucket=bucket)
    return MinioClientWrapper(cfg)

def _load_all_metadata_from_minio(client: MinioClientWrapper, bucket_prefix_bronze: str, video_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    metas = {}
    for vid in video_ids:
        key = f"{bucket_prefix_bronze}/{vid}/metadata.json"
        try:
            text = client.get_object_text(key)
            if text:
                metas[vid] = json.loads(text)
        except Exception as e:
            logger.warning(f"Could not load metadata for {vid}: {e}")
    return metas

def run_preprocess_full() -> None:
    paths_cfg = _load_paths_config()

    base_dir = Path(paths_cfg.get("data_root", "tiktok-data"))
    bronze_name = paths_cfg.get("bronze_subdir", "bronze")
    silver_name = paths_cfg.get("silver_subdir", "silver")
    gold_name = paths_cfg.get("gold_subdir", "gold")

    use_minio = bool(paths_cfg.get("use_minio", False))
    upload_silver = bool(paths_cfg.get("upload_silver", False))
    upload_gold = bool(paths_cfg.get("upload_gold", False))
    minio_bucket = paths_cfg.get("minio_bucket")
    video_ids = paths_cfg.get("video_ids", [])

    logger.info(f"Config: base={base_dir}, minio={use_minio}, bucket={minio_bucket}")

    minio_client = _init_minio(paths_cfg) if use_minio else None

    # Auto-discover video IDs if needed
    if not video_ids and use_minio and minio_client:
        logger.info(f"Listing videos from MinIO {bronze_name}...")
        video_ids = minio_client.list_subdirs(bronze_name)
        logger.info(f"Found {len(video_ids)} videos.")
    
    if not video_ids:
        logger.warning("No video IDs found to process.")
        return

    # 1. Fit Scaler (Require MinIO for now as per logic, or local crawl needed)
    if minio_client:
        metas_map = _load_all_metadata_from_minio(minio_client, bronze_name, video_ids)
    else:
        # Fallback local load not implemented fully here for brevity, assuming MinIO
        metas_map = {} 
        pass 

    meta_pre = MetadataPreprocessor()
    if metas_map:
        meta_pre.fit_numeric_scaler(list(metas_map.values()))
    else:
        logger.warning("No metadata loaded to fit scaler. Preprocessing might fail or be unscaled.")

    # 2. Init Modules
    # Load from preprocess_config if needed, using defaults for now
    audio_extractor = AudioExtractor(sample_rate=16000)
    audio_encoder = Wav2Vec2AudioEncoder(model_name="nguyenvulebinh/wav2vec2-base-vietnamese-250h")
    frame_extractor = VideoFrameExtractor(num_frames=16, frame_size=(224, 224))
    frame_loader = VideoFrameLoader(num_frames=16, frame_size=(224, 224))
    video_encoder = TimeSformerVideoEncoder(model_name="facebook/timesformer-base-finetuned-k400")

    # 3. Setup Feature Builder & Saver
    feature_builder = MultimodalFeatureBuilder(
        base_dir=base_dir, 
        silver_name=silver_name, 
        gold_name=gold_name,
        bronze_name=bronze_name
    )
    feature_saver = FeatureSaver(
        base_dir=base_dir,
        gold_name=gold_name,
        use_minio=use_minio,
        upload_gold=upload_gold,
        minio_bucket=minio_bucket
    )
    
    collected_rows: List[MultimodalFeatureRow] = []

    # 4. Processing Loop
    for i, vid in enumerate(video_ids):
        logger.info(f"Processing {i+1}/{len(video_ids)}: {vid}")
        
        local_bronze_vid = base_dir / bronze_name / vid
        local_silver_vid = base_dir / silver_name / vid
        dst_meta_path = local_bronze_vid / "metadata.json"

        try:
            local_bronze_vid.mkdir(parents=True, exist_ok=True)
            local_silver_vid.mkdir(parents=True, exist_ok=True)

            # --- RESUME LOGIC ---
            # Check if Silver already exists on MinIO
            silver_exists = False
            if minio_client and minio_client.check_silver_exists(vid):
                logger.info(f"Silver exists for {vid}, downloading skipping extraction...")
                try:
                    minio_client.download_silver_files(vid, base_dir)
                    silver_exists = True
                except Exception as e:
                    logger.warning(f"Failed to download silver for {vid}, re-processing: {e}")
            
            if silver_exists:
                # SKIP EXTRACTION -> Jump to Row Build
                # Need metadata json for Row Build if not in silver
                pass 
            else:
                # --- EXTRACTION START ---
                # Download Bronze
                if minio_client:
                    minio_client.download_bronze_video(vid, base_dir) 
                
                # Audio
                dst_video_path = local_bronze_vid / "video.mp4"
                if not dst_video_path.exists():
                     logger.warning(f"Video file missing: {dst_video_path}")
                     continue

                audio_path = audio_extractor.extract_audio_for_video(dst_video_path, local_silver_vid, vid)
                if audio_path:
                    audio_emb = audio_encoder.encode_file(audio_path)
                    np.save(local_silver_vid / "audio_embedding.npy", audio_emb)
                
                # Video
                frames_dir = local_silver_vid / "frames"
                frames_dir.mkdir(parents=True, exist_ok=True)
                frame_extractor.extract_frames(dst_video_path, frames_dir, vid)
                frames_tensor = frame_loader.load_frames(frames_dir)
                video_emb = video_encoder.encode(frames_tensor)
                np.save(local_silver_vid / "video_embedding.npy", video_emb)

                # Metadata
                meta = metas_map.get(vid)
                if not meta and dst_meta_path.exists():
                    with open(dst_meta_path) as f: meta = json.load(f)
                
                if meta:
                    feats = meta_pre.transform_single(meta)
                    np.savez_compressed(local_silver_vid / "metadata_features.npz", **feats)
                
                # Upload Silver
                if minio_client and upload_silver:
                    minio_client.upload_silver_dir(vid, base_dir)
                # --- EXTRACTION END ---

            # --- BUILD ROW ---
            row = feature_builder.build_row_from_dir(vid, local_silver_vid, dst_meta_path)
            if row:
                collected_rows.append(row)
            
            logger.info(f"Success: {vid}")

        except Exception as e:
            logger.exception(f"Failed {vid}")
        
        finally:
            # Cleanup
            safe_rmtree(local_bronze_vid)
            safe_rmtree(local_silver_vid)

    # 5. Save Gold Dataset
    if collected_rows:
        logger.info(f"Saving {len(collected_rows)} rows to Gold...")
        out_path = feature_saver.save_npz("multimodal_dataset", collected_rows)
        logger.info(f"Dataset saved at: {out_path}")
    else:
        logger.warning("No rows collected. Gold dataset not created.")

if __name__ == "__main__":
    run_preprocess_full()
