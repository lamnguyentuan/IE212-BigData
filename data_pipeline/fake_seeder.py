
import numpy as np
import json
from pathlib import Path
import sys
import os

# Add root
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from common.utils.minio_utils import MinioConfig, MinioClientWrapper

def seed_fake_data(num_videos=5):
    bucket_name = os.getenv("MINIO_BUCKET", "tiktok-data")
    cfg = MinioConfig.from_env(bucket=bucket_name)
    mc = MinioClientWrapper(cfg)
    
    print(f"Seeding {num_videos} fake videos to bucket '{cfg.bucket}'...")
    
    for i in range(num_videos):
        vid = f"fake_video_{i}"
        print(f"Generating {vid}...")
        
        # Bronze: metadata
        meta = {
            "id": vid,
            "desc": f"This is a simulated video {i}",
            "author": "sim_user",
            "stats": {"diggCount": 100 + i, "shareCount": i}
        }
        
        # Create tmp files
        os.makedirs("tmp_seed", exist_ok=True)
        with open("tmp_seed/metadata.json", "w") as f:
            json.dump(meta, f)
            
        with open("tmp_seed/video.mp4", "wb") as f:
            f.write(b"fake_video_content")
            
        # Upload Bronze
        mc.upload_file("tmp_seed/metadata.json", f"bronze/{vid}/metadata.json")
        mc.upload_file("tmp_seed/video.mp4", f"bronze/{vid}/video.mp4")
        
        # Silver: features
        # Audio embedding (e.g. 768 dim)
        audio_emb = np.random.rand(768).astype(np.float32)
        np.save("tmp_seed/audio_embedding.npy", audio_emb)
        
        # Video embedding (e.g. 768 dim)
        video_emb = np.random.rand(768).astype(np.float32)
        np.save("tmp_seed/video_embedding.npy", video_emb)
        
        # Metadata features (e.g. 50 dim)
        meta_feat = np.random.rand(50).astype(np.float32)
        np.savez("tmp_seed/metadata_features.npz", features=meta_feat)
        
        # Upload Silver
        mc.upload_file("tmp_seed/audio_embedding.npy", f"silver/{vid}/audio_embedding.npy")
        mc.upload_file("tmp_seed/video_embedding.npy", f"silver/{vid}/video_embedding.npy")
        mc.upload_file("tmp_seed/metadata_features.npz", f"silver/{vid}/metadata_features.npz")
        
    print("Done seeding.")

if __name__ == "__main__":
    seed_fake_data()
