# 🚀 Project Execution Guide

This document provides a step-by-step guide to running the entire **TikTok Harmful Content Detection System**, from data ingestion to real-time serving.

---

## 🛠️ 0. Prerequisites

- **OS**: Linux
- **Docker & Docker Compose**: Installed and running.
- **Python**: 3.9+
- **Conda** (Recommended): Create an environment.
  ```bash
  conda create -n tiktok-env python=3.9 -y
  conda activate tiktok-env
  pip install -r requirements.txt
  ```

---

## 🏗️ 1. Infrastructure Setup

Start the core services (MinIO, Kafka, Spark, MongoDB, Zookeeper).

```bash
docker-compose up -d
```

**Verify Services:**
- MinIO Console: [http://localhost:9001](http://localhost:9001) (User/Pass: `minioadmin` / `minioadmin`)
- Spark Master: [http://localhost:8080](http://localhost:8080)
- MongoDB: `localhost:27017`

---

## 📥 2. Data Ingestion

Acquire data into the **Bronze Layer** (MinIO).

### Option A: Crawl TikTok
```bash
# Crawl by hashtag
python data-ingestion/tiktok_crawl/tiktok_scraper.py --hashtag "vietnam" --limit 20
```

### Option B: Upload TikHarm Dataset
```bash
python data-ingestion/tikharm_upload/upload_tikharm_to_minio.py
```

---

## ⚙️ 3. Batch ETL Pipeline

Process raw data from Bronze → Silver → Gold layers using Spark.

```bash
python data-pipeline/spark-streaming/main_stream.py --mode batch
```
*This handles data cleaning, resizing, and preparation for training.*

---

## 🧠 4. Offline Training

Train the Multimodal Model.

### Step 4.1: Feature Extraction
Extract features (TimeSformer, Wav2Vec2, BERT) from videos in Silver layer.
```bash
python offline_training/preprocessing/pipelines/preprocess_full_pipeline.py
```

### Step 4.2: Pretrain (TikHarm)
Train the base model on the 4-class TikHarm dataset.
```bash
python offline_training/pretrain/train_tikharm.py
```
*Artifacts saved to `offline_training/artifacts/pretrain` and pushed to HF Hub.*

### Step 4.3: Finetune (VN Dataset)
Finetune the model for binary classification (Safe vs Not Safe).
```bash
python offline_training/finetune/train_tiktok_vn.py
```
*Artifacts saved to `offline_training/artifacts/finetune` and pushed to HF Hub.*

---

## ⚡ 5. Real-Time Streaming & Serving

Set up the live detection loop.

### Step 5.1: Start Model Serving API
Start the FastAPI server (loads the finetuned model).
```bash
# Run locally (or use docker container 'model-serving')
uvicorn model-serving.app.server:app --port 8000 --reload
```

### Step 5.2: Start Spark Streaming Job
Consume events from Kafka, call the Model API, and write to MongoDB.
```bash
python data-pipeline/spark-streaming/main_stream.py --mode stream
```

### Step 5.3: Trigger Data (Simulate Ingestion)
Send a video event to Kafka to test the stream.
```bash
python data-pipeline/kafka/producer/send_video_events.py
```

---

## 📊 6. Dashboard

Monitor the system in real-time.

```bash
streamlit run dashboard/app.py
```
**Access**: [http://localhost:8501](http://localhost:8501)

---

## 🧹 Cleanup
To stop all infrastructure:
```bash
docker-compose down
```
