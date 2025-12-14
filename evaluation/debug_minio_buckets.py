import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from common.utils.minio_utils import MinioConfig, MinioClientWrapper
import logging

logging.basicConfig(level=logging.INFO)

def list_bucket_objects(bucket_name):
    print(f"--- Listing objects in bucket: {bucket_name} ---")
    try:
        cfg = MinioConfig.from_env(bucket=bucket_name)
        mc = MinioClientWrapper(cfg)
        
        # Access underlying client
        objects = mc._client.list_objects(bucket_name, recursive=True)
        
        video_ids = set()
        count_files = 0
        silver_files = 0
        gold_files = 0
        for obj in objects:
            count_files += 1
            # bronze/{video_id}/video.mp4 or silver/{video_id}/video_embedding.npy
            parts = obj.object_name.split('/')
            if len(parts) >= 2 and parts[0] == 'bronze':
                video_ids.add(parts[1])
            if len(parts) >= 2 and parts[0] == 'silver':
                silver_files += 1
            if len(parts) >= 2 and parts[0] == 'gold':
                gold_files += 1
                if obj.object_name.endswith('.npz'):
                    print(f"  FOUND GOLD DATASET: {obj.object_name} ({obj.size/1024/1024:.2f} MB)")
        
        print(f"  Total Objects: {count_files}")
        print(f"  Unique Video IDs (in bronze/): {len(video_ids)}")
        print(f"  Files in silver/: {silver_files}")
        print(f"  Files in gold/: {gold_files}")
            
    except Exception as e:
        print(f"Error listing bucket {bucket_name}: {e}")
    print("\n")

if __name__ == "__main__":
    list_bucket_objects("tiktok-data")
    list_bucket_objects("tikharm")
