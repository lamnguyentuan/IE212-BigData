"""
Spark Session Helper.

Provides a unified way to create SparkSession onto MinIO.
"""

import os
from pathlib import Path
from typing import Optional
from pyspark.sql import SparkSession
from common.utils.config_loader import load_yaml

ROOT = Path(__file__).resolve().parents[2]

def get_spark_session(app_name: str = "TikTokApp", config_path: Optional[str] = None) -> SparkSession:
    """
    Create a SparkSession with MinIO configuration.
    """
    if config_path is None:
        config_path = str(ROOT / "data_pipeline/spark-streaming/spark_config.yaml")

    conf_dict = {}
    try:
        data = load_yaml(config_path)
        spark_cfg = data.get("spark", {})
        
        # Override app name
        final_app_name = app_name or spark_cfg.get("app_name", "TikTokApp")
        
        # Hadoop/MinIO conf
        hadoop_conf = spark_cfg.get("hadoop_conf", {})
        
        # Allow env var overrides for secrets
        if os.getenv("MINIO_ENDPOINT"):
             hadoop_conf["fs.s3a.endpoint"] = os.getenv("MINIO_ENDPOINT")
        if os.getenv("MINIO_ACCESS_KEY"):
             hadoop_conf["fs.s3a.access.key"] = os.getenv("MINIO_ACCESS_KEY")
        if os.getenv("MINIO_SECRET_KEY"):
             hadoop_conf["fs.s3a.secret.key"] = os.getenv("MINIO_SECRET_KEY")

        builder = SparkSession.builder.appName(final_app_name)
        
        if spark_cfg.get("master"):
            builder = builder.master(spark_cfg["master"])
            
        # Apply hadoop conf via spark.hadoop prefix
        for k, v in hadoop_conf.items():
            builder = builder.config(f"spark.hadoop.{k}", v)

        # Apply generic spark conf
        spark_ops = spark_cfg.get("spark_conf", {})
        for k, v in spark_ops.items():
            builder = builder.config(k, v)

        # Basic delta/parquet/avro support if needed
        # builder = builder.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") ...
            
        spark = builder.getOrCreate()
        return spark
        
    except Exception as e:
        print(f"Error loading spark config: {e}. Returning default session.")
        return SparkSession.builder.appName(app_name).getOrCreate()
