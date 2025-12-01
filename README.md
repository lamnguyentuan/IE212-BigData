# **TikTok Harmful Content Detection – Multimodal + Big Data Pipeline**
---

## 🧩 **1. Giới thiệu dự án**

Hệ thống phát hiện video độc hại trên TikTok với khả năng:

* Thu thập video TikTok theo thời gian thực
* Trích xuất đa phương thức (video frames, audio, OCR, ASR, comments)
* Huấn luyện mô hình đa mô thức:

  * **TimeSformer** (video)
  * **wav2vec2** (audio)
  * **ViSoBERT** (text)
  * **Cross-Attention Fusion**
* Phân loại Safe / Not Safe
* Triển khai trong pipeline Big Data: Kafka → Spark → MinIO → MongoDB
* Dashboard giám sát real-time

**Storage chính:** MinIO (S3-compatible)
**Tổ chức dữ liệu:** Medallion (Bronze → Silver → Gold)

---

## 🏛 **2. Tổng quan kiến trúc**

```
Crawl → Kafka → Spark Streaming → MinIO (Bronze → Silver → Gold)
                           ↓
                     Model Serving
                           ↓
                       MongoDB
                           ↓
                   Streamlit Dashboard
```

---

# 📦 **3. Cấu trúc thư mục dự án (FULL TREE)**

```
tiktok-harmful-content-detection/
│
├── README.md
├── requirements.txt
├── docker-compose.yml
├── .env
│
├── offline-training/
│   ├── data_access/
│   │   ├── minio_reader.py
│   │   └── dataset_config.yaml
│   ├── datasets/                     # optional (local debug)
│   ├── preprocessing/
│   │   ├── extract_frames.py
│   │   ├── extract_audio.py
│   │   ├── ocr_text.py
│   │   ├── asr_text.py
│   │   └── clean_text.py
│   ├── models/
│   │   ├── timesformer_encoder.py
│   │   ├── wav2vec_encoder.py
│   │   ├── viso_bert_encoder.py
│   │   ├── fusion_cross_attention.py
│   │   └── classifier.py
│   ├── pretrain/
│   │   ├── pretrain_config.yaml
│   │   └── pretrain_run.py
│   ├── finetune/
│   │   ├── finetune_config.yaml
│   │   └── finetune_run.py
│   ├── utils/
│   │   ├── dataset_loader.py
│   │   ├── metrics.py
│   │   └── scheduler.py
│   └── artifacts/
│       ├── pretrained_model.pt
│       ├── finetuned_model.pt
│       └── tokenizer/
│
├── data-pipeline/
│   ├── kafka/
│   │   ├── producer/
│   │   │   ├── crawl_and_push.py
│   │   │   └── send_metadata.py
│   │   ├── consumer/
│   │   │   └── read_stream.py
│   │   ├── topics/
│   │   │   ├── video-topic
│   │   │   └── metadata-topic
│   │   └── kafka_config.json
│   │
│   ├── storage/
│   │   ├── minio_client.py
│   │   ├── minio_config.yaml
│   │   └── medallion_layout.md
│   │
│   ├── medallion/
│   │   ├── bronze_loader/
│   │   │   ├── save_raw_to_minio.py
│   │   │   └── validate_raw.py
│   │   ├── silver_transform/
│   │   │   ├── clean_text_job.py
│   │   │   ├── process_media_job.py
│   │   │   └── write_silver_minio.py
│   │   └── gold_curate/
│   │       ├── build_training_sets.py
│   │       ├── build_analytics_views.py
│   │       └── write_gold_minio.py
│   │
│   ├── spark-streaming/
│   │   ├── main_stream.py
│   │   ├── preprocess/
│   │   │   ├── ffmpeg_ops.py
│   │   │   ├── ocr_engine.py
│   │   │   ├── asr_engine.py
│   │   │   └── text_processing.py
│   │   ├── inference/
│   │   │   └── call_serving.py
│   │   └── spark_config.yaml
│   │
│   └── utils/
│       ├── logger.py
│       └── helpers.py
│
├── model-serving/
│   ├── app/
│   │   ├── server.py
│   │   ├── load_model.py
│   │   ├── inference.py
│   │   ├── schemas.py
│   │   └── requirements.txt
│   ├── artifacts/
│   │   └── finetuned_model.pt
│   └── Dockerfile
│
├── dashboard/
│   ├── app.py
│   ├── components/
│   │   ├── video_player.py
│   │   ├── charts.py
│   │   └── stats_box.py
│   ├── services/
│   │   ├── mongodb_query.py
│   │   └── minio_reader.py
│   └── styles/
│       └── theme.css
│
└── deployment/
    ├── airflow/
    │   ├── dags/
    │   │   ├── bronze_loader_dag.py
    │   │   ├── silver_transform_dag.py
    │   │   └── gold_curate_dag.py
    │   └── airflow_config.cfg
    │
    ├── docker/
    │   ├── kafka/
    │   ├── spark/
    │   ├── minio/
    │   │   ├── Dockerfile
    │   │   └── minio.env.example
    │   ├── mongodb/
    │   └── other_services/
    │
    ├── configs/
    │   ├── spark-submit.sh
    │   └── environment.yaml
    │
    └── k8s/
        ├── kafka.yaml
        ├── spark.yaml
        ├── minio.yaml
        ├── serving.yaml
        └── dashboard.yaml
```

---

# 🏗 **4. Chi tiết các thư mục & nhiệm vụ**

---

## 🔥 **offline-training/**

Huấn luyện mô hình:

* Pretrain trên TikHarm (4 classes)
* Finetune trên dataset Việt Nam (Safe / Not Safe)

Đọc dữ liệu từ MinIO: `gold/training_sets/...`

---

## 🚚 **data-pipeline/**

### **📌 1. kafka/**

Nhận sự kiện crawl → đẩy vào pipeline.

### **📌 2. storage/**

Client MinIO + config + layout Medallion.

### **📌 3. medallion/**

3 tầng xử lý:

#### **Bronze → raw ingestion**

* video gốc
* audio gốc
* metadata thô
* OCR/ASR thô

#### **Silver → cleaned + processed**

* video resized
* audio normalized
* text OCR/ASR đã clean
* features sơ cấp

#### **Gold → curated + ML-ready**

* dataset train/val/test
* analytics views
* inference-ready views

### **📌 4. spark-streaming/**

Spark đọc Kafka → đọc/ghi MinIO theo từng layer.

---

## 🧠 **model-serving/**

FastAPI + PyTorch:

* Load mô hình finetune
* Nhận embedding từ Spark
* Trả nhãn Safe / Not Safe

---

## 📊 **dashboard/**

Realtime monitoring:

* MongoDB → thống kê kết quả infer
* MinIO (gold/analytics_views) → biểu đồ phân tích nội dung

---

## 🚀 **deployment/**

Airflow + Docker.

---

# 🗂 **5. MinIO Medallion Layout**

```
tiktok-data/
│
├── bronze/
│   ├── video/
│   ├── audio/
│   ├── ocr_raw/
│   ├── asr_raw/
│   └── metadata_raw/
│
├── silver/
│   ├── video_processed/
│   ├── audio_processed/
│   ├── text_clean/
│   ├── comments_clean/
│   └── features_base/
│
└── gold/
    ├── training_sets/
    │   ├── tikharm_4class/
    │   └── vn_safe_notsafe/
    ├── inference_views/
    └── analytics_views/
```

---

# 🎯 **6. Hướng dẫn chạy nhanh**

```
docker-compose up -d
python data-pipeline/kafka/producer/crawl_and_push.py
spark-submit data-pipeline/spark-streaming/main_stream.py
uvicorn model-serving/app/server:app
streamlit run dashboard/app.py
```

