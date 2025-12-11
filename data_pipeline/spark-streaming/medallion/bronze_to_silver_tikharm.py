"""
TikHarm Bronze -> Silver ETL Job.

Reads raw metadata keys from MinIO Bronze and normalizes them into Silver Parquet tables.
"""

import sys
from pathlib import Path
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from data_pipeline.utils.spark_helper import get_spark_session
from data_pipeline.utils.logger import get_logger

logger = get_logger("etl-bronze-silver-tikharm")

def run_job():
    spark = get_spark_session("TikHarmBronzeToSilver")
    
    # TikHarm bucket
    # Note: Spark sees MinIO via s3a://<bucket>
    # Config is loaded in get_spark_session
    BRONZE_PATH = "s3a://tikharm/bronze/*/*.json"  # Reads all metadata.json
    SILVER_PATH = "s3a://tikharm/silver/metadata"

    schema = StructType([
        StructField("video_id", StringType(), True),
        StructField("split", StringType(), True),
        StructField("label_raw", StringType(), True),
        StructField("label", StringType(), True),
        StructField("original_filename", StringType(), True),
        StructField("source", StringType(), True)
    ])

    logger.info(f"Reading from {BRONZE_PATH}...")

    try:
        # Read JSON
        df = spark.read.schema(schema).json(BRONZE_PATH)
        
        # Add timestamp/processing info if needed
        df_silver = df.withColumn("ingested_at", F.current_timestamp()) \
                      .withColumn("bronze_video_path", 
                                  F.concat(F.lit("s3a://tikharm/bronze/"), F.col("video_id"), F.lit("/video.mp4")))
        
        logger.info(f"Writing to {SILVER_PATH}...")
        
        (
            df_silver.write
            .mode("overwrite")
            .parquet(SILVER_PATH)
        )
        
        logger.info("TikHarm Bronze -> Silver Success.")
        
    except Exception as e:
        logger.error(f"ETL Failed: {e}")
        # If path doesn't exist (no data yet), Spark might error.
        
    spark.stop()

if __name__ == "__main__":
    run_job()
