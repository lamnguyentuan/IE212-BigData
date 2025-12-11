# 🎥 TikTok Harmfulness Detection System (Real-time Big Data Pipeline)

**Hệ thống Phân tích & Phát hiện Nội dung Độc hại trên TikTok theo Thời gian thực**

![Badge](https://img.shields.io/badge/Status-Active-success)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![Spark](https://img.shields.io/badge/Apache%20Spark-Streaming-orange)

Đồ án môn học **IE212 - Big Data**, tập trung xây dựng một hệ thống xử lý dữ liệu lớn đa phương thức (Video, Audio, Metadata) End-to-End từ khâu thu thập đến hiển thị cảnh báo.

---

## 🏗️ Kiến trúc Hệ thống (System Architecture)

Hệ thống được thiết kế theo mô hình **Lambda Architecture** (tập trung vào Speed Layer cho Real-time Demo), bao gồm các thành phần chính:

1.  **Ingestion Layer (Thu thập)**
    *   **Crawler (Playwright + Python)**: Tự động tương tác với TikTok Web, tải video `.mp4`, trích xuất metadata (comment, view, like).
    *   **Apache Kafka**: Message Broker chịu tải cao, nhận sự kiện từ Crawler và phân phối đến bộ xử lý.

2.  **Storage Layer (Lưu trữ)**
    *   **MinIO (Data Lake)**: Lưu trữ dữ liệu phi cấu trúc (Unstructured Data) như Video, Audio, Features.
    *   **MongoDB (NoSQL)**: Lưu trữ dữ liệu cấu trúc (Structured Data) như kết quả dự đoán, metadata, logs.

3.  **Processing Layer (Xử lý)**
    *   **Apache Spark Structured Streaming**: Xử lý dữ liệu thời gian thực từ Kafka.
    *   **Multimodal Inference**:
        *   **Video**: Trích xuất Frames -> TimeSformer Model.
        *   **Audio**: Trích xuất Audio -> Wav2Vec2 Model.
        *   **Fusion**: Kết hợp đặc trưng để phân loại `Safe` vs `Harmful`.
    *   **Model Serving API**: Microservice (FastAPI+Uvicorn) cung cấp khả năng Inference độc lập.

4.  **Orchestration Layer (Điều phối)**
    *   **Apache Airflow**: Lập lịch tự động (Schedule) cho việc Crawling định kỳ hoặc Retrain model.

5.  **Presentation Layer (Hiển thị)**
    *   **Streamlit Dashboard**: Giao diện người dùng theo dõi Real-time, phát lại video, hiển thị biểu đồ thống kê.

---

## 🛠️ Yêu cầu Cài đặt (Prerequisites)

*   **Docker & Docker Compose** (Bắt buộc)
*   **Python 3.9+** (Nếu chạy script client)
*   RAM: Tối thiểu 8GB (Khuyến nghị 16GB do chạy Spark & DL Models)

---

## 🚀 Hướng dẫn Chạy (Quick Start)

### 1. Khởi động Hạ tầng (Infrastructure)

Tại thư mục gốc của dự án:

```bash
# Build images và khởi chạy services (Airflow, Spark, Kafka, MinIO, Dashboard...)
docker-compose up -d --build
```

Đợi khoảng 2-5 phút để các image được build và services khởi động hoàn tất.

### 2. Truy cập Giao diện Quản lý

Nếu chạy trên máy cục bộ (Localhost):

*   **Airflow**: [http://localhost:8081](http://localhost:8081) (Admin: `admin`/`admin`)
*   **Dashboard**: [http://localhost:8501](http://localhost:8501)
*   **Spark Master**: [http://localhost:8080](http://localhost:8080)
*   **MinIO Console**: [http://localhost:9001](http://localhost:9001) (User: `minioadmin`/`minioadmin`)

*(Nếu chạy trên Server/VPS, hãy sử dụng SSH Tunnel để forward các port này về máy cá nhân).*

### 3. Demo Kịch bản End-to-End

Để thấy dữ liệu chạy từ Crawler -> Dashboard, hãy làm theo các bước sau:

#### Bước 1: Thu thập Dữ liệu (Trigger Crawl)
Vào **Airflow UI** -> Kích hoạt DAG `tiktok_crawl_pipeline`.
*hoặc chạy thủ công trong container:*
```bash
docker exec crawler python data_pipeline/crawl_only.py review
```

#### Bước 2: Giả lập Luồng dữ liệu (Producer)
Script này sẽ quét MinIO và gửi thông báo "New Video" tới Kafka.
```bash
# Cần cài venv ở máy host để chạy script này
export MINIO_ENDPOINT="localhost:9009"
export MINIO_ACCESS_KEY="minioadmin"
export MINIO_SECRET_KEY="minioadmin"

python data_pipeline/producer_simulator.py
```

#### Bước 3: Xử lý & Dự đoán (Spark Streaming)
Khởi chạy Spark Job để lắng nghe Kafka và gọi Model AI.
```bash
export MINIO_ENDPOINT="localhost:9009"
export MINIO_ACCESS_KEY="minioadmin"
export MINIO_SECRET_KEY="minioadmin"
export MONGO_URI="mongodb://localhost:27017/"

python data_pipeline/spark-streaming/main_stream.py --mode stream
```

**Kết quả:** Mở Dashboard tại `localhost:8501`, bạn sẽ thấy các video mới xuất hiện liên tục cùng nhãn dự đoán (Harmful/Safe)!

---

## 📂 Cấu trúc Dự án

```
.
├── airflow/                 # Cấu hình & DAGs cho Airflow
├── common/                  # Modules dùng chung (MinIO client, Features utils)
├── dashboard/               # Mã nguồn Streamlit Dashboard (Dockerfile riêng)
├── data_pipeline/           # Các script xử lý dữ liệu chính
│   ├── producer_simulator.py  # Giả lập Kafka Producer
│   ├── crawl_only.py          # Script Crawl gọn nhẹ
│   └── spark-streaming/       # PySpark Streaming Job
├── demo-crawl.Dockerfile    # Dockerfile cho Crawler Service
├── docker-compose.yml       # File định nghĩa toàn bộ hạ tầng Docker
├── model-serving/           # API Server cho AI Model (FastAPI)
├── offline_training/        # Quy trình huấn luyện Model (Preprocessing)
└── requirements.txt         # Các thư viện phụ thuộc
```

---

## 👨‍💻 Tác giả

Đồ án được thực hiện bởi nhóm sinh viên **IE212 - UIT**.
Mọi góp ý xin gửi về [Issues](https://github.com/your-repo/issues).
