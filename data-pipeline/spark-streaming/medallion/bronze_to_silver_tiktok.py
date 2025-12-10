"""
TikTok Bronze -> Silver ETL Job.

Reads crawled metadata from MinIO Bronze and normalizes into Silver Parquet.
"""

import sys
from pathlib import Path
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType, ArrayType, MapType, DoubleType

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from data_pipeline.utils.spark_helper import get_spark_session
from data_pipeline.utils.logger import get_logger

logger = get_logger("etl-bronze-silver-tiktok")

def run_job():
    spark = get_spark_session("TikTokBronzeToSilver")
    
    # TikTok bucket
    BRONZE_PATH = "s3a://tiktok-data/bronze/*/*.json"
    SILVER_PATH = "s3a://tiktok-data/silver/metadata"
    
    # Complex schema for crawled data
    # Adapt this to match exactly what your scraper produces (tiktok_scraper.py)
    schema = StructType([
        StructField("video_id", StringType(), True),
        StructField("url", StringType(), True),
        StructField("timestamp", DoubleType(), True),
        StructField("source", StringType(), True),
        StructField("description", StringType(), True),
        # Assuming stats is a flattened map or struct? 
        # Scraper says: "stats": { "likes": ..., ... }
        # Let's try Map first or Struct if consistent
        StructField("stats", MapType(StringType(), StringType()), True),
        StructField("tags", ArrayType(StringType()), True)
    ])

    logger.info(f"Reading from {BRONZE_PATH}...")
    
    try:
        df = spark.read.schema(schema).json(BRONZE_PATH)

        # Cleanup & Transformations
        # e.g. Extract likes from map
        df_clean = df.withColumn("likes", F.col("stats").getItem("likes").cast("long")) \
                     .withColumn("comments_count", F.col("stats").getItem("comments")) \
                     .withColumn("shares", F.col("stats").getItem("shares")) \
                     .withColumn("processed_at", F.current_timestamp())

        logger.info(f"Writing to {SILVER_PATH}...")

        (
            df_clean.write
            .mode("append") # Use append for crawled data generally, or overwrite if batching full bronze
            .partitionBy("source")
            .parquet(SILVER_PATH)
        )
        
        logger.info("TikTok Bronze -> Silver Success.")

    except Exception as e:
        logger.error(f"ETL Failed: {e}")

    spark.stop()

if __name__ == "__main__":
    run_job()
