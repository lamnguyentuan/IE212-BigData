# Data Pipeline Module

This module contains the Big Data processing logic, primarily powered by **Apache Spark** (and PySpark). It handles both batch ETL (Medallion Architecture) and Real-time Streaming.

## Directory Structure

- **`kafka/`**: Kafka configuration and Producer/Consumer scripts.
- **`spark-streaming/`**: Core Spark jobs and logic.
    - **`medallion/`**: ETL scripts (Bronze -> Silver -> Gold).
    - **`main_stream.py`**: The driver entry point.

## Prerequisites
- Spark 3.5.0+ installed or available via Docker.
- `pyspark` python library.
- Access to MinIO (S3 compatible) and Kafka.
