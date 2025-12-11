import json
import logging
import sys
from pathlib import Path
import os
import numpy as np

# Thêm project root vào system path để import được modules
FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from offline_training.preprocessing.features.feature_saver import FeatureSaver
from common.utils.minio_utils import MinioConfig, MinioClientWrapper
from dotenv import load_dotenv

load_dotenv()

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("update-labels")

def parse_label_studio_file(json_path: Path) -> dict:
    """
    Parse file JSON export từ Label Studio.
    Return dictionary: {video_id (str): label_int (int)}
    
    Mapping:
    - SAFE -> 0
    - NOT SAFE -> 2 (Harmful)
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Label file not found: {json_path}")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    mapping = {
        "SAFE": 0,
        "NOT SAFE": 2
    }
    
    id_to_label = {}
    
    for item in data:
        # 1. Trích xuất Video ID từ đường dẫn S3
        # data["video"] = "s3://tiktok-data/bronze/7580626145759038738/video.mp4"
        s3_path = item.get("data", {}).get("video", "")
        if not s3_path:
            continue
            
        parts = s3_path.split("/")
        # Cấu trúc: s3://bucket/bronze/{id}/video.mp4
        # parts: ['s3:', '', 'bucket', 'bronze', '{id}', 'video.mp4']
        if len(parts) < 5:
            continue
            
        video_id = parts[-2]
        
        # 2. Trích xuất Label
        # annotations[0].result[0].value.choices[0]
        try:
            annotations = item.get("annotations", [])
            if not annotations:
                continue
            
            result = annotations[0].get("result", [])
            if not result:
                continue
                
            val = result[0].get("value", {})
            choices = val.get("choices", [])
            if not choices:
                continue
            
            label_str = choices[0] # "SAFE" or "NOT SAFE"
            
            if label_str in mapping:
                id_to_label[video_id] = mapping[label_str]
        except Exception as e:
            logger.warning(f"Error parsing item for video {video_id}: {e}")
            continue
            
    return id_to_label

def update_dataset_labels(label_file_path: str, data_root: str = "tiktok-data_local", minio_upload: bool = True):
    base_dir = Path(data_root)
    gold_dir = base_dir / "gold"
    dataset_path = gold_dir / "multimodal_dataset.npz"
    
    # 1. Parse File Label
    logger.info(f"Parsing label file: {label_file_path}")
    label_map = parse_label_studio_file(Path(label_file_path))
    logger.info(f"Found {len(label_map)} labeled videos in JSON file.")
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found at: {dataset_path}")
        return

    # 2. Load Dataset hiện tại
    logger.info(f"Loading dataset: {dataset_path}")
    data = np.load(dataset_path)
    
    # Extract arrays
    video_ids = data['video_ids']
    labels = data['labels']
    # Các trường khác giữ nguyên
    video_embs = data['video_embs']
    audio_embs = data['audio_embs']
    text_embs = data['text_embs']
    metadata_numeric = data['metadata_numeric']
    
    updated_count = 0
    total_samples = len(video_ids)
    
    # 3. Update Labels
    new_labels = labels.copy()
    
    for i, vid in enumerate(video_ids):
        # video_ids trong npz có thể là bytes hoặc str, đảm bảo convert về str
        vid_str = str(vid) if isinstance(vid, str) else vid.decode('utf-8')
        
        if vid_str in label_map:
            new_label = label_map[vid_str]
            if new_labels[i] != new_label:
                new_labels[i] = new_label
                updated_count += 1
    
    logger.info(f"Updated {updated_count}/{total_samples} samples with new labels.")
    
    # 4. Save Dataset mới (ghi đè)
    if updated_count > 0:
        np.savez_compressed(
            dataset_path,
            video_ids=video_ids,
            video_embs=video_embs,
            audio_embs=audio_embs,
            text_embs=text_embs,
            metadata_numeric=metadata_numeric,
            labels=new_labels
        )
        logger.info(f"Saved updated dataset to: {dataset_path}")
        
        # 5. Upload lên MinIO (Optional)
        if minio_upload:
            try:
                # Load MinIO Config (hardcoded or env)
                # Dùng lại MinIO Utils
                cfg = MinioConfig.from_env(bucket="tiktok-data")
                mc = MinioClientWrapper(cfg)
                
                object_name = "gold/multimodal_dataset.npz"
                mc.upload_file(dataset_path, object_name)
            except Exception as e:
                logger.error(f"Failed to upload to MinIO: {e}")
    else:
        logger.info("No labels changed. Skipping save.")

if __name__ == "__main__":
    # Đường dẫn hardcode cho tiện user, hoặc có thể dùng argparse
    LABEL_FILE = "offline_training/preprocessing/tiktok_label_done.json"
    DATA_ROOT = "tiktok-data_local" # Folder local mà pipeline đang chạy
    
    update_dataset_labels(LABEL_FILE, DATA_ROOT)
