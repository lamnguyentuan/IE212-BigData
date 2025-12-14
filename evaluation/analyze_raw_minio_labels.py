import sys
import json
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from common.utils.minio_utils import MinioConfig, MinioClientWrapper

def analyze_raw_labels():
    print("Connecting to MinIO 'tiktok-data'...")
    cfg = MinioConfig.from_env(bucket="tiktok-data")
    mc = MinioClientWrapper(cfg)
    
    # List all objects
    objects = mc._client.list_objects("tiktok-data", recursive=True)
    
    json_files = []
    for obj in objects:
        if obj.object_name.endswith("metadata.json"):
            json_files.append(obj.object_name)
            
    print(f"Found {len(json_files)} metadata files.")
    
    safe_count = 0
    harmful_count = 0
    unknown_count = 0
    
    sample_printed = False
    
    for i, meta_obj_name in enumerate(json_files):
        try:
            # Read content
            resp = mc._client.get_object("tiktok-data", meta_obj_name)
            content = resp.read()
            resp.close()
            resp.release_conn()
            
            meta = json.loads(content)
            
            if not sample_printed:
                print(f"Sample Metadata Content: {json.dumps(meta, indent=2)}")
                print(f"Keys: {list(meta.keys())}")
                sample_printed = True
            
            # Try to find label in common keys
            # 'label', 'labels', 'ground_truth', 'category', 'class'
            label = None
            for key in ['label', 'labels', 'ground_truth', 'violation', 'harmful']:
                if key in meta:
                    label = meta[key]
                    break
            
            # Heuristic mapping if label is raw text
            if label is None:
                unknown_count += 1
                continue
                
            label_str = str(label).lower()
            
            if label_str in ["safe", "0"]:
                safe_count += 1
            elif label_str in ["not safe", "harmful", "1", "2", "3"]:
                harmful_count += 1
            else:
                # Need to see what actual values are
                # Maybe user used specific keywords
                if "harassment" in label_str or "hate" in label_str or "self-harm" in label_str or "violence" in label_str:
                     harmful_count += 1
                else:
                     unknown_count += 1
                     print(f"Unknown label: {label}")

        except Exception as e:
            print(f"Error reading {meta_obj_name}: {e}")
            
    print("\n--- Statistics on Raw TikTok Data (554 subset) ---")
    print(f"Total Metadata Files: {len(json_files)}")
    print(f"Safe: {safe_count}")
    print(f"Not Safe (Harmful): {harmful_count}")
    print(f"Unknown/Unlabeled: {unknown_count}")

if __name__ == "__main__":
    analyze_raw_labels()
