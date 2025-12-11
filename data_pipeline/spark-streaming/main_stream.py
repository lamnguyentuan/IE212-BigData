"""
Main Driver for Data Pipeline.

Can be used to:
1. Run Streaming Job (Real-time)
2. Run Batch Medallion Jobs (Bronze -> Silver -> Gold)

Usage:
    python main_stream.py --mode batch
    python main_stream.py --mode stream
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
import sys
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data_pipeline.utils.logger import get_logger
from medallion import (
    bronze_to_silver_tikharm,
    bronze_to_silver_tiktok,
    silver_to_gold_training_sets
)

logger = get_logger("data-pipeline-main")

def run_batch_etl():
    logger.info("=== Starting Batch ETL: Bronze -> Silver (TikHarm) ===")
    bronze_to_silver_tikharm.run_job()
    
    logger.info("=== Starting Batch ETL: Bronze -> Silver (TikTok) ===")
    bronze_to_silver_tiktok.run_job()
    
    logger.info("=== Starting Batch ETL: Silver -> Gold ===")
    silver_to_gold_training_sets.run_job()
    
    logger.info("=== Batch ETL Complete ===")

import json
import requests
import datetime
from pyspark.sql.functions import from_json, col, struct
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, MapType
from pymongo import MongoClient
from data_pipeline.utils.spark_helper import get_spark_session

# Configuration
import os

# Configuration
KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "video_events")
MODEL_SERVING_URL = os.getenv("MODEL_SERVING_URL", "http://localhost:8000/predict")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "tiktok_harm_db")
MONGO_COLLECTION = "predictions"

def write_to_mongo(batch_df, batch_id):
    """
    ForeachBatch function to call model serving API and write to MongoDB.
    """
    if batch_df.isEmpty():
        return
        
    records = batch_df.collect()
    logger.info(f"Processing batch {batch_id} with {len(records)} records...")
    
    # Initialize Mongo Client (per batch or per partition is better often, but batch ok for low volume)
    # Ideally partition level, but batch level is simpler to demo
    try:
        mongo_client = MongoClient(MONGO_URI)
        db = mongo_client[MONGO_DB]
        collection = db[MONGO_COLLECTION]
        
        results_to_insert = []
        
        for row in records:
            # Call Model Serving API
            try:
                # Assuming row has fields from JSON
                # Check server.py schemas: requires "video_id", optional "use_minio"
                payload = {
                    "video_id": row.video_id,
                    "use_minio": True  # Assuming production flow fetches from MinIO
                }
                
                response = requests.post(MODEL_SERVING_URL, json=payload, timeout=5) # fast timeout
                
                if response.status_code == 200:
                    pred = response.json()
                    # Flatten for MongoDB
                    doc = {
                        "video_id": pred["video_id"],
                        "label": pred["label_name"],
                        "label_id": pred["label_id"],
                        "confidence": pred["confidence"],
                        "probabilities": pred["probabilities"],
                        "processing_time": pred["processing_time_ms"],
                        "source": row.source,
                        "raw_path": row.raw_path,
                        "ingested_at": datetime.datetime.now(),
                        "timestamp": row.timestamp
                    }
                    results_to_insert.append(doc)
                else:
                    logger.warning(f"Model Serving failed for {row.video_id}: {response.text}")
                    
            except Exception as e:
                logger.error(f"Error processing video {row.video_id}: {e}")

        if results_to_insert:
            collection.insert_many(results_to_insert)
            logger.info(f"Inserted {len(results_to_insert)} predictions to MongoDB.")
            
        mongo_client.close()
        
    except Exception as e:
        logger.error(f"MongoDB/Batch Error: {e}")

def run_streaming():
    logger.info("Starting Streaming Job...")
    
    spark = get_spark_session("TikTokStreamingApp")
    
    # Schema for Kafka Message
    schema = StructType([
        StructField("video_id", StringType()),
        StructField("source", StringType()),
        StructField("raw_path", StringType()),
        StructField("timestamp", DoubleType()),
        StructField("metadata", MapType(StringType(), StringType()))
    ])
    
    # Read Kafka
    df_kafka = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BROKERS) \
        .option("subscribe", TOPIC) \
        .option("startingOffsets", "earliest") \
        .load()
        
    # Parse JSON
    df_parsed = df_kafka.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")
    
    # Write Stream (ForeachBatch)
    query = df_parsed.writeStream \
        .foreachBatch(write_to_mongo) \
        .outputMode("append") \
        .start()
        
    query.awaitTermination()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["batch", "stream"], default="batch", help="Execution mode")
    args = parser.parse_args()

    if args.mode == "batch":
        run_batch_etl()
    else:
        run_streaming()
