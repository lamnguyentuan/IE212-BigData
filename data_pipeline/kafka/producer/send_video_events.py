"""
Kafka Producer for Video Events.

Sends video ingestion events to Kafka to trigger processing.
"""

import json
import time
import sys
from pathlib import Path
from kafka import KafkaProducer

# Paths
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data_pipeline.utils.logger import get_logger

logger = get_logger("kafka-producer")

def load_config():
    config_path = ROOT / "data-pipeline/kafka/kafka_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)

class VideoEventProducer:
    def __init__(self):
        self.config = load_config()
        self.producer = KafkaProducer(
            bootstrap_servers=self.config["bootstrap_servers"],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.topic = self.config["topic_name"]
        logger.info(f"Initialized Kafka Producer for topic: {self.topic}")

    def send_event(self, video_id: str, source: str, raw_path: str, metadata: dict = None):
        """
        Send a video ingestion event.
        """
        event = {
            "video_id": video_id,
            "source": source,
            "raw_path": raw_path,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        future = self.producer.send(self.topic, event)
        try:
            record_metadata = future.get(timeout=10)
            logger.info(f"Sent event for {video_id} to partition {record_metadata.partition} offset {record_metadata.offset}")
        except Exception as e:
            logger.error(f"Failed to send event for {video_id}: {e}")

    def close(self):
        self.producer.close()

if __name__ == "__main__":
    # Test Usage
    producer = VideoEventProducer()
    producer.send_event("test_vid_001", "manual_test", "/tmp/video.mp4", {"author": "funa21"})
    producer.close()
