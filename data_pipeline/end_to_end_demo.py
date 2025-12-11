
import asyncio
import os
import sys
import json
import time
import requests
import datetime
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Setup Root
FILE = Path(__file__).resolve()
# data-pipeline/end_to_end_demo.py -> parents[1] = ROOT
ROOT = FILE.parents[1]
sys.path.append(str(ROOT))

load_dotenv()

# Imports from project
from offline_training.preprocessing.audio.audio_extractor import AudioExtractor
from offline_training.preprocessing.audio.audio_encoder_wav2vec import Wav2Vec2AudioEncoder
from offline_training.preprocessing.video.video_frame_extractor import VideoFrameExtractor
from offline_training.preprocessing.video.video_loader import VideoFrameLoader
from offline_training.preprocessing.video.video_encoder_timesformer import TimeSformerVideoEncoder
from common.features.preprocessor import MetadataPreprocessor
from common.utils.minio_utils import MinioConfig, MinioClientWrapper
from common.features.multimodal_feature_builder import MultimodalFeatureBuilder

# Crawler Import (Assuming demo-crawl.py is in ROOT)
sys.path.append(str(ROOT)) 
# demo-crawl might need to be imported as module or just use exec/subprocess if it's a script
# But I can try to import the class if I fix sys.path
try:
    from demo_crawl import TikTokScraper # Need to rename demo-crawl.py to demo_crawl.py or use importlib
except ImportError:
    # If file is demo-crawl.py (dash), standard import fails.
    # use importlib
    import importlib.util
    spec = importlib.util.spec_from_file_location("demo_crawl", str(ROOT / "demo-crawl.py"))
    demo_crawl = importlib.util.module_from_spec(spec)
    sys.modules["demo_crawl"] = demo_crawl
    spec.loader.exec_module(demo_crawl)
    TikTokScraper = demo_crawl.TikTokScraper

# CONFIG
DATA_ROOT = ROOT / "tiktok-data_local"
MINIO_BUCKET = "tiktok-data"
MODEL_SERVING_URL = "http://localhost:8000/predict"
OUTPUT_FILE = ROOT / "data-pipeline/simulated_predictions.jsonl"

def init_preprocessors():
    print("Loading Preprocessing Models...")
    audio_ex = AudioExtractor(sample_rate=16000)
    audio_enc = Wav2Vec2AudioEncoder(model_name="nguyenvulebinh/wav2vec2-base-vietnamese-250h")
    frame_ex = VideoFrameExtractor(num_frames=16, frame_size=(224, 224))
    frame_loader = VideoFrameLoader(num_frames=16, frame_size=(224, 224))
    video_enc = TimeSformerVideoEncoder(model_name="facebook/timesformer-base-finetuned-k400")
    meta_pre = MetadataPreprocessor()
    # Dummy fit for scaler if needed or load - for now just rely on default or simple transform
    # In real prod we load saved scaler. For demo we skip fit or fit on empty? 
    # MetadataPreprocessor handles empty? check fit_numeric_scaler logic.
    # It sets scaler. if not fit, transform might fail. 
    # Let's fit on dummy data
    dummy_meta = [{"likes": "0", "comments": "0", "shares": "0", "views": "0", "description": ""}]
    meta_pre.fit_numeric_scaler(dummy_meta)
    
    return audio_ex, audio_enc, frame_ex, frame_loader, video_enc, meta_pre

def preprocess_single(vid, components):
    audio_ex, audio_enc, frame_ex, frame_loader, video_enc, meta_pre = components
    
    base_dir = DATA_ROOT
    bronze_dir = base_dir / "bronze" / vid
    silver_dir = base_dir / "silver" / vid
    
    bronze_dir.mkdir(parents=True, exist_ok=True)
    silver_dir.mkdir(parents=True, exist_ok=True)
    
    # 0. MinIO Client
    cfg = MinioConfig.from_env(bucket=MINIO_BUCKET)
    mc = MinioClientWrapper(cfg)
    
    print(f"[{vid}] Downloading Bronze...")
    mc.download_bronze_video(vid, base_dir)
    
    # 1. Process
    video_path = bronze_dir / "video.mp4"
    if not video_path.exists():
        print(f"[{vid}] Video not found!")
        return False

    print(f"[{vid}] Extracting Features...")
    # Audio
    audio_path = audio_ex.extract_audio_for_video(video_path, silver_dir, vid)
    if audio_path:
        audio_emb = audio_enc.encode_file(audio_path)
        np.save(silver_dir / "audio_embedding.npy", audio_emb)
        
    # Video
    frames_dir = silver_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_ex.extract_frames(video_path, frames_dir, vid)
    frames_tensor = frame_loader.load_frames(frames_dir)
    video_emb = video_enc.encode(frames_tensor)
    np.save(silver_dir / "video_embedding.npy", video_emb)
    
    # Meta
    meta_path = bronze_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f: meta = json.load(f)
        feats = meta_pre.transform_single(meta)
        np.savez_compressed(silver_dir / "metadata_features.npz", **feats)
        
    # Upload Silver
    print(f"[{vid}] Uploading Silver...")
    mc.upload_silver_dir(vid, base_dir)
    
    return True

async def run_demo(hashtag):
    # 1. Init
    components = init_preprocessors()
    
    # 2. Crawl
    print(f"=== Starting Crawl for #{hashtag} ===")
    scraper = TikTokScraper()
    # Scrape 1-2 videos
    results = await scraper.scrape_hashtag(hashtag, max_videos=1)
    
    processed_ids = []
    
    for info in results:
        # Extract ID from Url or use generated one?
        # demo-crawl logic: creates ID from url or time. 
        # But where is the ID? info dict might not have it explicitly if not added.
        # Wait, scrape_single_video saves to bronze/{id}/metadata.json
        # The result loop in scrape_hashtag calls _extract_video_info -> returns dict.
        # But demo-crawl saves it using `video_id`.
        # I need to know that `video_id`.
        # I should modify demo-crawl to include video_id in the returned dict?
        # Or I can list MinIO bronze new items?
        # Let's verify demo-crawl source again. 
        # It calls `self._save_json_to_minio(video_info, meta_object)`.
        # It adds `video_s3_path`. 
        # It does NOT add `video_id` to video_info dict explicitly.
        # I should probably derive valid ID from s3 path or just list `bronze` dir locally if scraped?
        # demo-crawl saves to MinIO. 
        # I can just check `downloaded_ids` file? or parse URL?
        url = info.get("videoUrl", "")
        try:
            vid = url.split("video/")[-1].split("?")[0]
        except:
            continue
            
        print(f"=== Preprocessing {vid} ===")
        success = preprocess_single(vid, components)
        
        if success:
            print(f"=== Inference {vid} ===")
            try:
                payload = {"video_id": vid, "use_minio": True}
                resp = requests.post(MODEL_SERVING_URL, json=payload)
                if resp.status_code == 200:
                    pred = resp.json()
                    print(f"Result: {pred['label_name']} ({pred['confidence']:.2f})")
                    
                    # Log to Dashboard file
                    doc = {
                        "video_id": pred["video_id"],
                        "label": pred["label_name"],
                        "label_id": pred["label_id"],
                        "confidence": pred["confidence"],
                        "probabilities": pred["probabilities"],
                        "processing_time_ms": pred["processing_time_ms"],
                        "timestamp": datetime.datetime.now().isoformat(),
                        "source": "end-to-end"
                    }
                    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(doc) + "\n")
                else:
                     print(f"Inference Failed: {resp.text}")
            except Exception as e:
                print(f"Inference Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        tag = sys.argv[1]
    else:
        tag = "vietnam" # Default
        
    asyncio.run(run_demo(tag))
