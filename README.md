# TikTok Harmful Content Detection System

This project implements an end-to-end Big Data pipeline for detecting harmful content on TikTok. It utilizes a multimodal approach, analyzing video frames (TimeSformer), audio (Wav2Vec2), and metadata/text (ViSoBERT) to classify videos as Safe or Not Safe (covering Adult, Harmful, Suicide, etc.).

The system features:
- **Crawling System**: Scrapes TikTok videos and uploads them to a Data Lake.
- **Medallion Architecture**: Organizes data in MinIO (Bronze/Silver/Gold) for scalable processing.
- **Offline Training**: Pretraining on the TikHarm dataset and Finetuning on Vietnamese TikTok data.
- **Real-time Processing**: Spark Streaming pipeline for processing new videos via Kafka.
- **Model Serving**: FastAPI service for real-time inference.
- **Dashboard**: Streamlit app for monitoring and analytics.

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed view of the system components and data flow.

## 🚀 Quickstart

### 1. Run Local Stack (Infrastructure)
Start the core services (MinIO, Kafka, Spark master, MongoDB, etc.) using Docker Compose:

```bash
docker-compose up -d
```
Access the services:
- **Dashboard**: http://localhost:8501
- **Airflow**: http://localhost:8081 (admin/admin)
- **MinIO Console**: http://localhost:9001 (minioadmin/minioadmin)
- **Spark Master**: http://localhost:8080
- **Model Serving**: http://localhost:8000/docs

### 2. Run Offline Preprocessing & Training
To prepare the data and train the models:

```bash
# 1. Run Preprocessing Pipeline (Extracts audio/video features)
python offline_training/preprocessing/pipelines/preprocess_full_pipeline.py

# 2. Pretrain on TikHarm dataset
python offline_training/pretrain/train_tikharm.py

# 3. Finetune on Vietnamese dataset
python offline_training/finetune/train_tiktok_vn.py
```

### 3. Run Streaming, Serving & Dashboard
To start the real-time detection flow:

```bash
# 1. Start Model Serving (FastAPI)
uvicorn model-serving.app.server:app --reload --port 8000

# 2. Start Spark Streaming Job
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
    data-pipeline/spark-streaming/main_stream.py

# 3. Start Dashboard
streamlit run dashboard/app.py
```

## 📂 Project Structure

- `data-ingestion/`: Crawler and data upload scripts.
- `data-pipeline/`: Kafka and Spark Streaming jobs.
- `offline_training/`: Preprocessing, Training, and Model definitions.
- `model-serving/`: Inference API.
- `dashboard/`: Monitoring UI.
- `deployment/`: Airflow DAGs and K8s configs.
