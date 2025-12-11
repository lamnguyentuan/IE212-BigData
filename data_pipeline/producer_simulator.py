import sys
import os
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import time
import json
import random
from kafka import KafkaProducer
from dotenv import load_dotenv

# Ensure we can import from common
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from common.utils.minio_utils import MinioConfig, MinioClientWrapper

load_dotenv()

KAFKA_BROKER = os.getenv("KAFKA_BROKERS", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "video_events")

def get_video_ids_from_minio():
    bucket_name = os.getenv("MINIO_BUCKET", "tiktok-data")
    cfg = MinioConfig.from_env(bucket=bucket_name)
    mc = MinioClientWrapper(cfg)
    # We use silver because we know features are there, but in real flow 
    # producer might come from bronze event. 
    # Let's list silver for simplicity as we know we have data there.
    try:
        subdirs = mc.list_subdirs("silver")
        return subdirs
    except Exception as e:
        print(f"Error listing MinIO: {e}")
        return []

def main():
    print(f"Connecting to Kafka at {KAFKA_BROKER}...")
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    except Exception as e:
        print(f"Failed to connect to Kafka: {e}")
        return

    print("Fetching video IDs from MinIO...")
    video_ids = get_video_ids_from_minio()
    print(f"Found {len(video_ids)} videos.")

    if not video_ids:
        print("No videos found to simulate.")
        return

    print(f"Start producing events to topic '{TOPIC}'...")
    
    # Loop indefinitely or just once? Let's loop a few times to simulate stream
    while True:
        vid = random.choice(video_ids)
        
        event = {
            "video_id": vid,
            "source": "simulator",
            "raw_path": f"s3://tiktok-data/bronze/{vid}/video.mp4",
            "timestamp": time.time(),
            "metadata": {
                "simulated": "true"
            }
        }
        
        producer.send(TOPIC, event)
        print(f"Sent event for {vid}")
        
        # Sleep random time
        time.sleep(random.uniform(1, 5))

if __name__ == "__main__":
    main()
