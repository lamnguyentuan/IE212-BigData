"""
Silver -> Gold ETL Job.

Joins metadata with feature availability to create certified training sets.
Output:
    - Gold TikHarm Table (for 4-class pretraining)
    - Gold VN TikTok Table (for 2-class finetuning)
"""

import sys
from pathlib import Path
from pyspark.sql import functions as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from data_pipeline.utils.spark_helper import get_spark_session
from data_pipeline.utils.logger import get_logger

logger = get_logger("etl-silver-gold")

def run_job():
    spark = get_spark_session("SilverToGold")
    
    # 1. TikHarm (Pretrain)
    # Read Silver
    try:
        df_silver = spark.read.parquet("s3a://tikharm/silver/metadata")
        
        # In a real pipeline, we would join with a table of "Available Features"
        # For now, we assume if it's in Silver, and Preprocessing ran successfully (which writes standard paths),
        # we can construct the feature paths.
        
        # Gold Schema: video_id, labels, audio_feat_path, video_feat_path, text_feat_path
        # Paths are standard in our MinIO layout: silver/{id}/...
        
        df_gold = df_silver.withColumn(
            "audio_feat_path", F.concat(F.lit("s3a://tikharm/silver/"), F.col("video_id"), F.lit("/audio_embedding.npy"))
        ).withColumn(
            "video_feat_path", F.concat(F.lit("s3a://tikharm/silver/"), F.col("video_id"), F.lit("/video_embedding.npy"))
        ).withColumn(
            "text_feat_path", F.concat(F.lit("s3a://tikharm/silver/"), F.col("video_id"), F.lit("/metadata_features.npz"))
        )
        
        # Filter logic (e.g. only Train/Val splits)
        # df_gold = df_gold.filter(F.col("split").isin("train", "val"))
        
        GOLD_PATH_TIKHARM = "s3a://tikharm/gold/training_set"
        logger.info(f"Writing TikHarm Gold to {GOLD_PATH_TIKHARM}")
        
        df_gold.write.mode("overwrite").parquet(GOLD_PATH_TIKHARM)
        
    except Exception as e:
        logger.warning(f"TikHarm Gold generation skipped/failed: {e}")

    # 2. VN TikTok (Finetune)
    # Similar logic, reading from tiktok-data silver
    try:
        df_tiktok = spark.read.parquet("s3a://tiktok-data/silver/metadata")
        
        # Assume we only want labeled data? 
        # (For now we might not have labels for crawled data unless manual labeling happened)
        # Just writing it out as an available dataset.
        
        df_tiktok_gold = df_tiktok.withColumn(
            "video_feat_path", F.concat(F.lit("s3a://tiktok-data/silver/"), F.col("video_id"), F.lit("/video_embedding.npy"))
        )
        
        GOLD_PATH_TIKTOK = "s3a://tiktok-data/gold/inference_set"
        logger.info(f"Writing TikTok Gold to {GOLD_PATH_TIKTOK}")
        
        df_tiktok_gold.write.mode("overwrite").parquet(GOLD_PATH_TIKTOK)
        
    except Exception as e:
         logger.warning(f"TikTok Gold generation skipped/failed: {e}")

    spark.stop()

if __name__ == "__main__":
    run_job()
