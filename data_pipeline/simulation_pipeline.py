
import os
import sys
import json
import time
import requests
import datetime
import random
from pathlib import Path
from dotenv import load_dotenv

# Path setup to import from common
# File is at: IE212-BigData/data-pipeline/simulation_pipeline.py
# ROOT is: IE212-BigData
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from common.utils.minio_utils import MinioConfig, MinioClientWrapper

load_dotenv()

# Config
MODEL_SERVING_URL = os.getenv("MODEL_SERVING_URL", "http://localhost:8000/predict")
OUTPUT_FILE = Path(__file__).parent / "simulated_predictions.jsonl"

def get_video_ids_from_minio():
    print("Fetching video list from MinIO...")
    try:
        cfg = MinioConfig.from_env(bucket="tiktok-data")
        mc = MinioClientWrapper(cfg)
        # We list 'silver' because we need features to be present for inference to work
        subdirs = mc.list_subdirs("silver")
        return subdirs
    except Exception as e:
        print(f"Error listing MinIO: {e}")
        return []

def simulate_pipeline():
    print("=== Starting Simulation Pipeline ===")
    video_ids = get_video_ids_from_minio()
    print(f"Found {len(video_ids)} videos in 'silver' folder.")
    
    if not video_ids:
        print("No videos found. Please ensure preprocessing (Stage 1-3) is done or Silver data exists.")
        return

    # Clear old output
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    print(f"Writing results to: {OUTPUT_FILE}")

    # Use a set to avoid processing same ID too close, but for stream sim we can repeat
    # Let's just loop through them once for this demo, or loop indefinitely
    
    count = 0
    # Process each video
    for vid in video_ids:
        count += 1
        print(f"[{count}/{len(video_ids)}] Processing {vid}...")
        
        # 1. Simulate Kafka Event (Just the ID is needed for API)
        payload = {
            "video_id": vid,
            "use_minio": True # Enable on-demand fetch
        }
        
        try:
            # 2. Call Model Serving
            start_time = time.time()
            response = requests.post(MODEL_SERVING_URL, json=payload, timeout=30)
            
            if response.status_code == 200:
                pred = response.json()
                
                # 3. Format result (Simulate Spark transformation)
                doc = {
                    "video_id": pred["video_id"],
                    "label": pred["label_name"],
                    "label_id": pred["label_id"],
                    "confidence": pred["confidence"],
                    "probabilities": pred["probabilities"],
                    "processing_time_ms": pred["processing_time_ms"],
                    "timestamp": datetime.datetime.now().isoformat(),
                    "source": "simulation"
                }
                
                # 4. Write to sink (File instead of Mongo)
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(doc) + "\n")
                
                print(f"  -> Success: {doc['label']} ({doc['confidence']:.2f})")
                
            else:
                print(f"  -> API Error {response.status_code}: {response.text}")

        except Exception as e:
            print(f"  -> Exception: {e}")
        
        # Simulate stream delay
        time.sleep(1)

    print("=== Simulation Complete ===")
    print(f"Check {OUTPUT_FILE} for results.")

if __name__ == "__main__":
    simulate_pipeline()
