# DEMO SCRIPT - TIKTOK HARMFUL CONTENT DETECTION

Hướng dẫn chạy demo tự động hoàn toàn từ crawl -> preprocessing -> inference -> dashboard.

## 1. Chuẩn bị môi trường

Mở 3 terminal riêng biệt tại thư mục dự án:
`cd /home/funalee/UIT/IE104/project/IE212-BigData`

### Terminal 1: Infrastructure & Services
Khởi động MinIO, Kafka, Mongo, Model Serving, Dashboard.

```bash
# 1. Tắt các container cũ để sạch sẽ
docker-compose down

# 2. Cấu hình Model Serving (Quan trọng: Bucket mới & Model Funa)
export MINIO_BUCKET="tiktok-realtime"
export MODEL_CHECKPOINT_PATH=""  # Để trống để auto-load từ HF hoặc local cache
export HF_HUB_REPO="funa21/tiktok-vn-finetune"

# 3. Khởi động
docker-compose up -d

# 4. Kiểm tra
docker ps
# Đảm bảo model-serving, kafka, minio, dashboard đều UP.
```

## 2. Terminal 2: Automated Pipeline (Orchestrator)
Script này sẽ ngồi canh MinIO. Hễ có video mới (Bronze) là tự động xử lý -> Silver -> Gold -> Kafka.

```bash
# Export bucket name
export MINIO_BUCKET="tiktok-realtime"

# Cài đặt thư viện cần thiết (Chỉ chạy lần đầu)
pip install -r data_pipeline/requirements-auto.txt

# Chạy pipeline tự động
python3 data_pipeline/auto_pipeline.py
```
*Chờ đến khi thấy log: `👀 Start watching MinIO for new Bronze videos...`*

## 3. Terminal 3: Spark Inference & Dashboard
Khởi động Spark Streaming để lắng nghe Kafka và đẩy kết quả ra Dashboard.

```bash
# Export config
export MINIO_BUCKET="tiktok-realtime"
export KAFKA_TOPIC="video_events"

# Submit Spark Job
# Lưu ý: --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.2 (đã có trong container hoặc submit local)
# Nếu chạy local:
python data_pipeline/spark-streaming/main_stream.py --mode stream
```

*Nếu chạy bằng Docker (đã up ở bước 1):*
```bash
docker-compose exec spark-master python /app/data_pipeline/spark-streaming/main_stream.py --mode stream
```

## 4. Chạy Demo (Terminal 4 hoặc Terminal hiện tại)
Bây giờ mọi thứ đã sẵn sàng. Hãy crawl một video mới.

```bash
export MINIO_BUCKET="tiktok-realtime"
# Crawl hashtag #review (để crawl ít video demo)
python data_pipeline/crawl_only.py review
```

## Quy trình tự động sẽ diễn ra như sau:
1.  **Crawler**: Tải video -> Upload lên MinIO `tiktok-realtime/bronze`.
2.  **Auto Pipeline** (Terminal 2):
    - Phát hiện video mới.
    - Tải về local.
    - Trích xuất Audio (Wav2Vec2), Video (TimeSformer), Metadata.
    - Save `silver` (features).
    - Save `gold` (dataset row).
    - **Bắn tin nhắn sang Kafka**.
3.  **Spark** (Terminal 3):
    - Nhận tin nhắn từ Kafka.
    - Gọi API Model Serving (localhost:8000).
    - Model Serving tải feature từ MinIO (nếu cần) hoặc nhận vector.
    - Trả về kết quả (Safe/Harmful).
    - Spark lưu vào MongoDB.
4.  **Dashboard**:
    - Truy cập: http://localhost:8501
    - Dữ liệu mới sẽ tự động hiển thị (Refresh nếu cần).

## Troubleshooting
- **Lỗi model not found**: Kiểm tra log `docker logs model-serving`. Đảm bảo nó đã tải được model `funa21`.
- **Lỗi Kafka**: Đảm bảo `auto_pipeline.py` in ra "Event sent to Kafka".
