import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dotenv import load_dotenv
load_dotenv()

# Import Kafka
from kafka import KafkaProducer

# Import Preprocessing Modules (Reuse existing logic)
from offline_training.preprocessing.audio.audio_extractor import AudioExtractor
from offline_training.preprocessing.audio.audio_encoder_wav2vec import Wav2Vec2AudioEncoder
from offline_training.preprocessing.video.video_frame_extractor import VideoFrameExtractor
from offline_training.preprocessing.video.video_loader import VideoFrameLoader
from offline_training.preprocessing.video.video_encoder_timesformer import TimeSformerVideoEncoder
from common.features.preprocessor import MetadataPreprocessor
from common.utils.minio_utils import MinioConfig, MinioClientWrapper
from common.features.multimodal_feature_builder import MultimodalFeatureBuilder
from offline_training.preprocessing.features.feature_saver import FeatureSaver
from common.utils.file_io import safe_rmtree

# --- Configuration ---
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "tiktok-realtime") 
KAFKA_BROKER = os.getenv("KAFKA_BROKERS", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "video_events")

# Local directories for processing
DATA_ROOT = ROOT / "tikharm_local" # Use a separate local dir to avoid conflict with big datasets
DATA_ROOT.mkdir(exist_ok=True)

logger = logging.getLogger("auto-pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [AUTO-PIPE] %(message)s"
)

class AutoPipeline:
    def __init__(self):
        self._init_minio()
        self._init_kafka()
        self._init_processors()
        
        self.processed_videos = set()
        
        # Initial scan to avoid reprocessing everything on restart (optional)
        # For demo, we might want to process new things mostly.
        # Let's populate processed_videos from Silver folder in MinIO to be safe/efficient
        self._sync_state()

    def _init_minio(self):
        logger.info(f"Connecting to MinIO bucket: {MINIO_BUCKET}...")
        self.cfg = MinioConfig.from_env(bucket=MINIO_BUCKET)
        self.mc = MinioClientWrapper(self.cfg)
        
        # Ensure Gold/Silver exist
        # MinioWrapper creates bucket if missing, but subdirs are implicit
        
    def _init_kafka(self):
        logger.info(f"Connecting to Kafka: {KAFKA_BROKER}...")
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
        except Exception as e:
            logger.error(f"Kafka connection failed: {e}")
            self.producer = None

    def _init_processors(self):
        logger.info("Initializing Models & Processors (This may take a moment)...")
        self.audio_extractor = AudioExtractor(sample_rate=16000)
        self.audio_encoder = Wav2Vec2AudioEncoder(model_name="nguyenvulebinh/wav2vec2-base-vietnamese-250h")
        self.frame_extractor = VideoFrameExtractor(num_frames=16, frame_size=(224, 224))
        self.frame_loader = VideoFrameLoader(num_frames=16, frame_size=(224, 224))
        self.video_encoder = TimeSformerVideoEncoder(model_name="facebook/timesformer-base-finetuned-k400")
        self.meta_pre = MetadataPreprocessor()
        
        # We need to fit the numeric scaler. Ideally load a saved scaler.
        # For demo, we can fit on global stats or dummy fit if no data.
        # Let's try to fit on existing bronze data?
        # Or just use a default fit (might be slightly off but works for demo flow)
        # In a real system, we load a pickled scaler. 
        # Here we will try to load some metadata from MinIO to fit.
        logger.info("Fitting scaler on sample data...")
        self._fit_scaler()

        self.feature_builder = MultimodalFeatureBuilder(
            base_dir=DATA_ROOT,
            silver_name="silver",
            gold_name="gold",
            bronze_name="bronze"
        )
        
        # FeatureSaver logic for Gold
        self.feature_saver = FeatureSaver(
            base_dir=DATA_ROOT,
            gold_name="gold",
            use_minio=True,
            upload_gold=True,
            minio_bucket=MINIO_BUCKET
        )

    def _fit_scaler(self):
        # Quick hack: list some bronze metadata to fit
        try:
            video_ids = self.mc.list_subdirs("bronze")[:50] # Take 50 samples
            metas = []
            for vid in video_ids:
                try:
                    txt = self.mc.get_object_text(f"bronze/{vid}/metadata.json")
                    if txt:
                        metas.append(json.loads(txt))
                except:
                    pass
            
            if metas:
                self.meta_pre.fit_numeric_scaler(metas)
                logger.info(f"Scaler fitted on {len(metas)} samples.")
            else:
                logger.warning("No data to fit scaler. Features might be unnormalized.")
        except Exception as e:
            logger.error(f"Scaler fit error: {e}")

    def _sync_state(self):
        # List silver folder to know what's done
        try:
            silvers = self.mc.list_subdirs("silver")
            self.processed_videos = set(silvers)
            logger.info(f"Synced state: {len(self.processed_videos)} videos already in Silver.")
        except Exception as e:
            logger.warning(f"State sync failed: {e}")

    def start_watching(self, interval=5):
        logger.info("👀 Start watching MinIO for new Bronze videos...")
        while True:
            try:
                # 1. List Bronze
                bronze_ids = set(self.mc.list_subdirs("bronze"))
                
                # 2. Identify New
                new_ids = bronze_ids - self.processed_videos
                
                if new_ids:
                    logger.info(f"Found {len(new_ids)} new videos: {new_ids}")
                    for vid in new_ids:
                        self.process_video(vid)
                        self.processed_videos.add(vid)
                
                time.sleep(interval)
            except KeyboardInterrupt:
                logger.info("Stopping...")
                break
            except Exception as e:
                logger.error(f"Watch loop error: {e}")
                time.sleep(interval)

    def process_video(self, vid: str):
        logger.info(f"=== Processing {vid} ===")
        
        local_bronze = DATA_ROOT / "bronze" / vid
        local_silver = DATA_ROOT / "silver" / vid
        
        try:
            local_bronze.mkdir(parents=True, exist_ok=True)
            local_silver.mkdir(parents=True, exist_ok=True)

            # 1. Download Bronze (Video + Meta)
            self.mc.download_bronze_video(vid, DATA_ROOT)
            
            video_path = local_bronze / "video.mp4"
            meta_path = local_bronze / "metadata.json"
            
            if not video_path.exists():
                logger.warning(f"Video {vid} missing video.mp4, skipping.")
                return

            # 2. Extract Features (Audio, Video, Meta) -> Silver
            
            # Audio
            audio_path = self.audio_extractor.extract_audio_for_video(video_path, local_silver, vid)
            if audio_path:
                audio_emb = self.audio_encoder.encode_file(audio_path)
                np.save(local_silver / "audio_embedding.npy", audio_emb)
            
            # Video
            frames_dir = local_silver / "frames"
            frames_dir.mkdir(exist_ok=True)
            self.frame_extractor.extract_frames(video_path, frames_dir, vid)
            frames_tensor = self.frame_loader.load_frames(frames_dir)
            video_emb = self.video_encoder.encode(frames_tensor)
            np.save(local_silver / "video_embedding.npy", video_emb)
            
            # Metadata
            if meta_path.exists():
                with open(meta_path) as f: meta = json.load(f)
                feats = self.meta_pre.transform_single(meta)
                np.savez_compressed(local_silver / "metadata_features.npz", **feats)
            
            # Upload Silver to MinIO
            logger.info("Uploading Silver...")
            self.mc.upload_silver_dir(vid, DATA_ROOT)
            
            # 3. Build Gold Row & Upload
            logger.info("Building Gold...")
            row = self.feature_builder.build_row_from_dir(vid, local_silver, meta_path)
            
            if row:
                # Custom save to Gold (FeatureSaver usually saves full dataset, we save single row here)
                # We can reuse save_npz passing list of 1
                self.feature_saver.save_npz(f"gold_{vid}", [row])
                logger.info("Gold uploaded.")
            
            # 4. Trigger Kafka
            if self.producer:
                event = {
                    "video_id": vid,
                    "source": "auto_pipeline",
                    "timestamp": time.time(),
                    "bucket": MINIO_BUCKET,
                    "status": "silver_gold_ready"
                }
                self.producer.send(TOPIC, event)
                self.producer.flush()
                logger.info(f"Event sent to Kafka topic '{TOPIC}'")

            logger.info(f"✅ Finished {vid}")

        except Exception as e:
            logger.error(f"Failed to process {vid}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Cleanup local disk
            safe_rmtree(local_bronze)
            safe_rmtree(local_silver)
            # Cleanup gold temp file if needed? FeatureSaver cleans? No, but okay for demo.
            safe_rmtree(DATA_ROOT / "gold")

if __name__ == "__main__":
    app = AutoPipeline()
    app.start_watching()
