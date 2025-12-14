import sys
import numpy as np
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from common.utils.minio_utils import MinioConfig, MinioClientWrapper

def download_and_check():
    print("--- Downloading Gold Dataset from tiktok-data ---")
    cfg = MinioConfig.from_env(bucket="tiktok-data")
    mc = MinioClientWrapper(cfg)
    
    remote_path = "gold/multimodal_dataset.npz"
    local_path = ROOT / "evaluation/dataset_tiktok_real.npz"
    
    try:
        mc.download_object(remote_path, local_path)
        print(f"Downloaded to {local_path}")
    except Exception as e:
        print(f"Download failed: {e}")
        return

    # Check content
    try:
        data = np.load(local_path, allow_pickle=True)
        print(f"Keys: {list(data.keys())}")
        
        # Check shapes
        vid_ids = data['video_ids']
        labels = data['labels']
        video_embs = data['video_embs']
        
        print(f"Total Samples: {len(labels)}")
        print(f"Video feature shape: {video_embs.shape}")
        
        # Compare with manual labels
        manual_json = ROOT / "offline_training/preprocessing/tiktok_label_done.json"
        with open(manual_json, 'r') as f:
            manual_data = json.load(f)
            
        print(f"Manual Labels Count: {len(manual_data)}")
        
        # Build map of manual labels
        manual_map = {} # video_id -> label (0: Safe, 1: Harmful)
        for item in manual_data:
            # Extract ID from S3 path: s3://tiktok-data/bronze/{id}/video.mp4
            s3_path = item.get('data', {}).get('video', '')
            if 'bronze/' in s3_path:
                vid_id = s3_path.split('bronze/')[1].split('/')[0]
                
                # Get label
                anns = item.get('annotations', [])
                if anns and anns[0].get('result'):
                    val = anns[0]['result'][0]['value']['choices'][0]
                    lbl = 0 if val == "SAFE" else 1
                    manual_map[vid_id] = lbl
        
        print(f"Mapped {len(manual_map)} manual labels.")
        
        # Check consistency
        match = 0
        mismatch = 0
        missing = 0
        
        updated_labels = []
        
        for i, vid_id in enumerate(vid_ids):
            # vid_id might be simple string or derived.
            # In FeatureSaver, it stored whatever ID was passed.
            # Typically ID string.
            vid_str = str(vid_id)
            if vid_str in manual_map:
                man_lbl = manual_map[vid_str]
                current_lbl = labels[i]
                if man_lbl == current_lbl:
                    match += 1
                else:
                    mismatch += 1
                updated_labels.append(man_lbl)
            else:
                missing += 1
                updated_labels.append(labels[i]) # Keep original if unknown?
                
        print(f"Matches: {match}")
        print(f"Mismatches: {mismatch}")
        print(f"Missing in Manual: {missing}")
        
        # Save updated dataset with verified labels
        if mismatch > 0 or missing == 0:
            print("creating dataset_tiktok_real_verified.npz with manual labels...")
            np.savez_compressed(
                ROOT / "evaluation/dataset_tiktok_real_verified.npz",
                video_ids=vid_ids,
                video_embs=video_embs,
                audio_embs=data['audio_embs'],
                text_embs=data['text_embs'],
                metadata_numeric=data['metadata_numeric'],
                labels=np.array(updated_labels)
            )
            print("Saved verified dataset.")

    except Exception as e:
        print(f"Error reading npz: {e}")

if __name__ == "__main__":
    download_and_check()
