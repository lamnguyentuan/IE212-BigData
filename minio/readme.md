# 🗂 **README – Thư mục `minio/`**

Thư mục `minio/` chứa toàn bộ cấu hình và module Python dùng để kết nối và thao tác với MinIO – hệ thống object storage cho toàn bộ dự án.

MinIO được sử dụng như một **Data Lake** nơi lưu trữ dữ liệu ở 3 tầng **Medallion Architecture**:

* **Bronze** (raw data)
* **Silver** (processed data)
* **Gold** (curated ML-ready data)

Dự án có 2 bộ dữ liệu độc lập:

* **TikTok dataset** (VN raw crawl → dùng để Fine-tune)
* **TikHarm dataset** (dataset 27GB → dùng để Pretrain)

Vì vậy, hệ thống sử dụng **2 bucket riêng biệt**, mỗi bucket đều có 3 tầng bronze/silver/gold.

---

## 📁 **Cấu trúc thư mục**

```
minio/
├── README.md
├── config_tiktok.yaml        # cấu hình MinIO cho bucket "tiktok-data"
├── config_tikharm.yaml       # cấu hình MinIO cho bucket "tikharm"
└── minio_client.py           # hàm tạo MinIO client dùng chung
```

---

# 📝 **Giải thích từng file**

## 1. `config_tiktok.yaml`

Chứa thông tin cấu hình cho bucket **tiktok-data**, dùng để lưu dữ liệu crawl TikTok theo Medallion:

```
tiktok-data/
├── bronze/tiktok/
├── silver/tiktok/
└── gold/tiktok/
```

Nội dung ví dụ:

```yaml
endpoint: "localhost:9000"
access_key: "minioadmin"
secret_key: "minioadmin123"
secure: false
bucket: "tiktok-data"
```

### Dùng cho:

* Upload dữ liệu TikTok crawl (video, metadata, OCR, ASR)
* Spark đọc & ghi dữ liệu TikTok
* Fine-tune model

---

## 2. `config_tikharm.yaml`

Chứa thông tin cấu hình MinIO cho bucket **tikharm-data**, dùng để lưu dataset TikHarm (27GB):

```
tikharm-data/
├── bronze/tikharm/
├── silver/tikharm/
└── gold/tikharm/
```

Ví dụ:

```yaml
endpoint: "localhost:9000"
access_key: "minioadmin"
secret_key: "minioadmin123"
secure: false
bucket: "tikharm"
```

### Dùng cho:

* Upload TikHarm dataset vào bronze
* Spark xử lý TikHarm (silver & gold)
* Pretrain model

---

## 3. `minio_client.py`

Module Python dùng để:

* Tạo MinIO client từ file config YAML
* Kiểm tra bucket tồn tại (nếu không thì tự động tạo)
* Trả về `(client, bucket_name)` để các module khác dùng

### Nội dung chính:

```python
from minio import Minio
import yaml
from pathlib import Path

BASE = Path(__file__).parent

def get_minio_client(config_name: str = "config_tiktok.yaml"):
    cfg_path = BASE / config_name
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    client = Minio(
        cfg["endpoint"],
        access_key=cfg["access_key"],
        secret_key=cfg["secret_key"],
        secure=cfg["secure"]
    )

    bucket = cfg["bucket"]
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

    return client, bucket
```

### Cách sử dụng:

```python
from minio_client import get_minio_client

client, bucket = get_minio_client("config_tiktok.yaml")

client.fput_object(
    bucket_name=bucket,
    object_name="bronze/tiktok/video/123456.mp4",
    file_path="local_data/123456.mp4",
)
```

---

# 🧭 **Thêm bucket mới (nếu cần)**

1. Tạo file YAML mới
   Ví dụ: `config_experiment.yaml`

2. Thêm nội dung:

```yaml
endpoint: "localhost:9000"
access_key: "minioadmin"
secret_key: "minioadmin123"
secure: false
bucket: "experiment-data"
```

3. Dùng:

```python
client, bucket = get_minio_client("config_experiment.yaml")
```

---

# 🔥 **Liên kết với thư mục khác trong dự án**

### Dữ liệu từ TikTok crawl

`data-ingestion/tiktok_crawl/` → upload → bucket `tiktok-data`

Ví dụ:

```
tiktok-data/bronze/tiktok/video
tiktok-data/bronze/tiktok/metadata_raw
```

### Dữ liệu TikHarm 27GB

`data-ingestion/tikharm_upload/` → upload → bucket `tikharm-data`

Ví dụ:

```
tikharm-data/bronze/tikharm/video
tikharm-data/bronze/tikharm/metadata_raw
```

### Spark sẽ đọc + ghi:

```
s3a://tiktok-data/bronze/tiktok/...
s3a://tiktok-data/silver/tiktok/...
s3a://tikharm-data/gold/tikharm/...
```

---

# 📌 **Lời khuyên sử dụng**

* Tách bucket theo dataset là đúng → dễ quản lý, dễ training, dễ backup.
* Mỗi bucket đều có bronze/silver/gold → không cần gộp vào 1 bucket phức tạp.
* `minio_client.py` xử lý cả 2 bucket thông qua 2 file config.
* Spark config chỉ việc đọc `config_tiktok.yaml` hoặc `config_tikharm.yaml` tùy job.
