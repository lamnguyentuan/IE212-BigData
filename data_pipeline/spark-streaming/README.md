# Spark Streaming & ETL

All Spark jobs are executed from this directory.

## 1. Batch ETL (The Medallion Pipeline)
We use Spark to move data through the Medallion layers.

### Run Batch ETL
```bash
python main_stream.py --mode batch
```
This runs the following sequence:
1.  **Bronze -> Silver (TikHarm)**: Cleans raw TikHarm uploads (`bronze_to_silver_tikharm.py`).
2.  **Bronze -> Silver (TikTok)**: Cleans raw Crawler uploads (`bronze_to_silver_tiktok.py`).
3.  **Silver -> Gold**: Aggregates features into training datasets and analytics views (`silver_to_gold_training_sets.py`).

## 2. Real-Time Streaming
The pipeline consumes real-time video ingestion events from Kafka.

### Run Streaming Job
```bash
python main_stream.py --mode stream
```
**Logic**:
- Reads from Kafka topic `video_events`.
- Calls the **Model Serving API** (HTTP) for each micro-batch.
- Writes predictions to **MongoDB**.
