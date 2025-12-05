# ***Tổng quan data preprocessing***
Dưới đây là **README.md hoàn chỉnh** dành cho thư mục:

```
offline-training/preprocessing/
```

README này:

* Giải thích **toàn bộ kiến trúc** (audio, video, metadata, features, pipelines, utils).
* Giải thích **Bronze → Silver → Gold** theo Medallion Architecture.
* Giải thích **Config** và **cách chạy full pipeline**.

Pipeline này thực hiện toàn bộ bước **tiền xử lý dữ liệu Offline** cho hệ thống phân tích nội dung TikTok gồm đa phương thức:

* **Video** → frame sampling + TimeSformer embedding
* **Audio** → ffmpeg extraction + Wav2Vec2 embedding
* **Text (caption + comments)** → normalization, PhoBERT embedding
* **Metadata** → numeric features + text features + toxicity/constructiveness
* **Fused multimodal embedding** → concat(audio + video + text + metadata)

Dữ liệu được tổ chức theo **Medallion Architecture**:

```
tiktok-data/
│
├── bronze/       # dữ liệu thô 100%
│     └── {video_id}/video.mp4 + metadata.json
│
├── silver/       # dữ liệu đã preprocess
│     └── {video_id}/frames/, audio.wav, metadata_features.npz ...
│
└── gold/         # dataset multimodal cuối cùng (.npz)
      └── multimodal_dataset.npz
```

---

# 📦 1. Cấu trúc thư mục preprocessing

```
offline-training/preprocessing/
│
├── audio/                    # Xử lý audio (extract, wav2vec2 encoder)
│   ├── audio_extractor.py
│   ├── audio_encoder_wav2vec.py
│   └── audio_utils.py
│
├── video/                    # Xử lý video (frame extractor, TimeSformer encoder)
│   ├── video_frame_extractor.py
│   ├── video_loader.py
│   └── video_encoder_timesformer.py
│
├── text/                     # OCR / ASR (optional, chưa bật mặc định)
│
├── metadata/                 # Xử lý metadata từ TikTok
│   ├── embeddings.py         # PhoBERT encoder, ViNormT5
│   ├── classifiers.py        # Toxic / Constructive
│   ├── numeric_features.py   # likes, shares, ratios
│   ├── date_features.py      # age_days, cyclic month
│   ├── comments.py           # flatten tree, comment stats
│   ├── preprocessor.py       # MetadataPreprocessor tổng hợp
│   └── metadata_preprocess.py
│
├── features/                 # Build multimodal feature rows
│   ├── feature_schema.py
│   ├── feature_saver.py
│   └── multimodal_feature_builder.py
│
├── pipelines/                # Chạy từng bước của pipeline
│   ├── preprocess_audio_pipeline.py
│   ├── preprocess_video_pipeline.py
│   ├── preprocess_metadata_pipeline.py
│   └── build_multimodal_dataset.py
│
├── utils/                    # Công cụ dùng chung
│   ├── logging_utils.py
│   ├── timer.py
│   ├── file_io.py
│   ├── minio_utils.py
│   └── constants.py
│
└── configs/
    ├── paths.yaml
    ├── preprocess_config.yaml
    ├── encoders.yaml
    ├── metadata_preprocess.yaml
    └── text_models.yaml
```

---

# 🪨 2. Bronze / Silver / Gold (Medallion Architecture)

### 🔶 **BRONZE — Raw layer**

```
tiktok-data/bronze/{video_id}/
│── video.mp4
└── metadata.json (caption, comments, stats…)
```

Không chỉnh sửa gì.

---

### ⚪ **SILVER — Preprocessed layer**

```
tiktok-data/silver/{video_id}/
│── frames/frame_0001.jpg ...
│── audio.wav
│── metadata_features.npz
│── caption.txt, comments.txt, merged_text.txt
└── clean_metadata.json
```

Bao gồm:

* Video → frames
* Audio → 16kHz mono wav
* Metadata → PhoBERT embedding, numeric features, toxicity/constructiveness

---

### 🟡 **GOLD — Feature store**

```
tiktok-data/gold/
└── multimodal_dataset.npz
```

`multimodal_dataset.npz` chứa:

* `video_emb`
* `audio_emb`
* `text_emb`
* `metadata_numeric`
* `fused` (concat tất cả modality)
* `labels` (nếu có)

---

# ⚙️ 3. Cấu hình trong `configs/`

📌 `paths.yaml`

```yaml
data_root: "tiktok-data"
bronze_subdir: "bronze"
silver_subdir: "silver"
gold_subdir: "gold"

use_minio: false          # bật / tắt đồng bộ MinIO
upload_silver: false      # upload silver/* lên MinIO sau khi preprocess
minio_bucket: "tiktok-data"

video_ids: []             # nếu rỗng → tự scan bronze/*
```

---

📌 `preprocess_config.yaml`

```yaml
audio_sample_rate: 16000
num_frames: 16
frame_size: [224, 224]
```

---

📌 `encoders.yaml`

```yaml
audio:
  model_name: "facebook/wav2vec2-base"

video:
  model_name: "facebook/timesformer-base-finetuned-k400"
```

---

📌 `metadata_preprocess.yaml`

```yaml
text_model_name: "vinai/phobert-base-v2"
toxicity_model_name: "funa21/phobert-finetuned-victsd-toxic-v2"
construct_model_name: "funa21/phobert-finetuned-victsd-constructiveness-v2"
norm_model_name: "meoo225/ViNormT5"

reference_date: "2025-12-05"
assume_year: 2025

max_desc_len: 128
max_tags_len: 64
max_comments_len: 256
```

---

# 🚀 4. Chạy toàn bộ pipeline

Chạy theo thứ tự:

---

## **1️⃣ Preprocess audio**

```bash
python -m offline_training.preprocessing.pipelines.preprocess_audio_pipeline
```

Sinh ra:

```
silver/{video_id}/audio.wav
silver/{video_id}/audio_embedding.npy
```

---

## **2️⃣ Preprocess video**

```bash
python -m offline_training.preprocessing.pipelines.preprocess_video_pipeline
```

Sinh ra:

```
silver/{video_id}/frames/
silver/{video_id}/video_embedding.npy
```

---

## **3️⃣ Preprocess metadata**

```bash
python -m offline_training.preprocessing.pipelines.preprocess_metadata_pipeline
```

Sinh ra:

```
silver/{video_id}/metadata_features.npz
```

---

## **4️⃣ Build multimodal dataset (Gold)**

```bash
python -m offline_training.preprocessing.pipelines.build_multimodal_dataset
```

Sinh ra:

```
gold/multimodal_dataset.npz
```

---

# 🗂️ 5. Nội dung của multimodal dataset (.npz)

File `gold/multimodal_dataset.npz` gồm:

| Key                | Mô tả                             |
| ------------------ | --------------------------------- |
| `video_ids`        | Danh sách video                   |
| `video_emb`        | TimeSformer embedding             |
| `audio_emb`        | Wav2Vec2 embedding                |
| `text_emb`         | comments_emb hoặc desc_emb        |
| `metadata_numeric` | numeric_scaled đầy đủ 25+ feature |
| `fused`            | vector concat tất cả modality     |
| `labels`           | -1 nếu chưa có nhãn               |

Nạp dataset:

```python
import numpy as np

data = np.load("tiktok-data/gold/multimodal_dataset.npz")
x = data["fused"]      # shape (N, D_total)
y = data["labels"]     # shape (N,)
```

---

# 🔗 6. MinIO integration (optional)

Bật trong `paths.yaml`:

```yaml
use_minio: true
upload_silver: true
minio_bucket: "tiktok-data"
```

Cài MinIO SDK:

```bash
pip install minio
```

Set biến môi trường:

```bash
export MINIO_ENDPOINT=localhost:9000
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin
export MINIO_SECURE=0   # http
```

---

# 📌 7. Lưu đồ Preprocessing Pipeline

```
               ┌────────────┐
               │  BRONZE    │
               │video.mp4   │
               │metadata.json
               └──────┬─────┘
                      │
        ┌─────────────┼────────────┐
        │             │             │
        ▼             ▼             ▼
┌────────────┐ ┌────────────┐ ┌───────────────┐
│Audio Extract│ │Frame Extract│ │Metadata Process│
│Wav2Vec2     │ │TimeSformer   │ │PhoBERT, stats │
└──────┬─────┘ └──────┬──────┘ └────────┬────────┘
       │               │                │
       └───────────────┼────────────────┘
                       ▼
              ┌────────────────┐
              │    SILVER      │
              │audio/frames/   │
              │metadata_features
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │     GOLD       │
              │ fused dataset  │
              │ multimodal.npz │
              └────────────────┘
```

---

# 🎉 8. Kết luận

Pipeline này tách rời thành:

* **Module xử lý chuyên biệt** (audio/video/metadata/features)
* **Module orchestration (pipelines)**
* **Config YAML linh hoạt**
* **Hỗ trợ MinIO** (optional)
* **Khớp hoàn toàn với Medallion Architecture**

Bạn có thể:

* Plug-n-play chạy offline
* Tích hợp vào Airflow / Prefect
* Dùng dataset `.npz` để huấn luyện mô hình đa phương thức

---

Nếu bạn muốn, mình có thể viết tiếp:

* README cho `offline-training/pretrain/`
* Sơ đồ kiến trúc model như hình minh họa
* Cách huấn luyện model classifier từ `gold/multimodal_dataset.npz`

Chỉ cần nói mình biết nhé!



# ***Cấu trúc thư mục trên minIO***

## 🟫 **BRONZE (Raw Layer — Dữ liệu thô nguyên bản)**

```
tiktok-data/
└── bronze/
    └── {video_id}/
        ├── video.mp4
        └── metadata.json
```

Ví dụ:

```
tiktok-data/bronze/7219231231231/video.mp4
tiktok-data/bronze/7219231231231/metadata.json
```

---

## ⚪ **SILVER (Preprocessed Layer — Dữ liệu đã xử lý sơ cấp)**

Ở Silver, **vẫn giữ cấu trúc theo video_id** để dễ trace back toàn bộ pipeline.

```
tiktok-data/
└── silver/
    └── {video_id}/
        ├── frames/
        │     ├── frame_0001.jpg
        │     ├── frame_0002.jpg
        │     └── ...
        │
        ├── audio.wav                     # audio đã chuẩn hoá 16 kHz
        │
        ├── caption.txt                   # caption đã cleaned
        ├── comments.txt                  # comments đã cleaned
        ├── merged_text.txt               # caption + comments
        │
        └── clean_metadata.json           # metadata đã chuẩn hoá
```

Ví dụ:

```
tiktok-data/silver/7219231231231/frames/frame_0001.jpg
tiktok-data/silver/7219231231231/audio.wav
tiktok-data/silver/7219231231231/merged_text.txt
```

---

## 🟡 **GOLD (Feature Layer — Dữ liệu đã trích embedding)**

Vẫn theo từng video_id, nhưng chia file theo modality:

```
tiktok-data/
└── gold/
    └── {video_id}/
        ├── video_embedding.npy
        ├── audio_embedding.npy
        ├── text_embedding.npy
        ├── fused_embedding.npy           # concat(video + audio + text)
        │
        └── label.json                    # Safe / Not Safe
```

Ví dụ:

```
tiktok-data/gold/7219231231231/video_embedding.npy
tiktok-data/gold/7219231231231/fused_embedding.npy
tiktok-data/gold/7219231231231/label.json
```

---

# 🎯 **Tóm tắt đầy đủ cấu trúc MinIO theo yêu cầu**

```
tiktok-data/
├── bronze/
│    └── {video_id}/
│         ├── video.mp4
│         └── metadata.json
│
├── silver/
│    └── {video_id}/
│         ├── frames/
│         │     ├── frame_0001.jpg
│         │     └── ...
│         ├── audio.wav
│         ├── caption.txt
│         ├── comments.txt
│         ├── merged_text.txt
│         └── clean_metadata.json
│
└── gold/
     └── {video_id}/
          ├── video_embedding.npy
          ├── audio_embedding.npy
          ├── text_embedding.npy
          ├── fused_embedding.npy
          └── label.json
```

# ***Cấu trúc thư mục data processing***

Thư mục `offline-training/preprocessing/` chứa toàn bộ code tiền xử lý dữ liệu để chuẩn bị cho bước **offline training mô hình đa phương thức (video + audio + text + metadata)** trong dự án TikHarm.

Mục tiêu chính:

- Chuẩn hoá metadata TikTok (likes, comments, shares, hashtags, date, …)
- Chuẩn hoá văn bản bằng **ViNormT5**
- Phát hiện **Toxicity** và **Constructiveness** trong comment tree
- Sinh **text embeddings** (Phobert) cho description, tags, comments
- Trích xuất **video features** (frame / clip embedding)
- Trích xuất **audio features** (wav2vec, …)
- Kết hợp tất cả thành **multimodal feature** lưu ra Silver/Gold layer (Parquet/JSON/NPY/MinIO)

---

## 🗂 Cấu trúc thư mục

```text
offline-training/
└── preprocessing/
    ├── metadata/
    ├── video/
    ├── audio/
    ├── text/
    ├── features/
    ├── utils/
    ├── pipelines/
    └── notebooks/
```

---

## 1️⃣ `metadata/` — Xử lý metadata TikTok

> Các file này chịu trách nhiệm đọc JSON metadata (như ví dụ bạn đưa), chuẩn hoá số, text, comment, và sinh metadata embedding.

**Cấu trúc (dự kiến):**

```text
metadata/
├── __init__.py
├── metadata_preprocessor.py      # class MetadataPreprocessor (dùng ViNormT5 + Phobert + Toxic/Constructive)
├── numeric_features.py           # parse K/M, log1p, ratio, date encoding
├── text_normalizer.py            # wrapper model meoo225/ViNormT5
├── toxicity_classifier.py        # wrapper funa21/phobert-finetuned-victsd-toxic-v2
├── constructive_classifier.py    # wrapper funa21/phobert-finetuned-victsd-constructiveness-v2
├── text_embeddings.py            # Phobert-base embedding cho desc/tags/comments
├── comments_processing.py        # flatten comments_tree, basic stats
└── metadata_schema.py            # optional: validate/normalize input JSON
```

**Main entry:**

* `MetadataPreprocessor` trong `metadata_preprocessor.py`:

  * Chuẩn hoá các trường số: likes, comments, shares, bookmarks, views
  * Tạo các feature tỉ lệ: like_rate, engagement_rate, …
  * Date → `age_days`, `month_sin`, `month_cos`
  * Chuẩn hoá text (description, tags, comments) bằng **ViNormT5**
  * Map emoji → token (`<EMOJI_LAUGH>`, …)
  * Flatten `comments_tree`, tính:

    * số comment, avg length, phần trăm cmt có emoji cười, có dấu hỏi
    * số lượng & tỉ lệ **toxic comments**
    * số lượng & tỉ lệ **constructive comments**
  * Encode description, tags, comments bằng Phobert → embedding
  * Log + scale numeric feature (StandardScaler)
  * Trả về:

    * `numeric_scaled`, `numeric_raw`
    * `desc_emb`, `tags_emb`, `comments_emb`

---

## 2️⃣ `video/` — Xử lý video

> Chịu trách nhiệm load video, trích xuất frame, và (sau này) tạo video embedding.

**Cấu trúc gợi ý:**

```text
video/
├── __init__.py
├── video_loader.py               # đọc video từ local hoặc MinIO
├── video_frame_extractor.py      # ffmpeg/decord/pyav → frames
├── video_encoder_timesformer.py  # (sau này) TimeSformer/ViViT embedding
└── video_utils.py                # helper: resize, clip, fps, etc.
```

Output mong muốn:

* Tập frame/clip (để debug)
* Hoặc vector `video_emb` (np.ndarray / torch.Tensor) cho mỗi video ID

---

## 3️⃣ `audio/` — Xử lý audio từ video

> Extract audio track, chuẩn hoá, sinh audio embedding.

**Cấu trúc gợi ý:**

```text
audio/
├── __init__.py
├── audio_extractor.py            # ffmpeg: .mp4 → .wav
├── audio_encoder_wav2vec.py      # wav2vec2 / hubert / XLSR embedding
└── audio_utils.py                # resample, mono, chunk, etc.
```

Output:

* File `.wav` / `.flac` trung gian (tuỳ bạn)
* Vector `audio_emb` (per video)

---

## 4️⃣ `text/` — OCR / ASR / text cleaning

> Dùng nếu bạn trích text từ video (subtitles, speech, text overlay).

**Cấu trúc gợi ý:**

```text
text/
├── __init__.py
├── ocr_processor.py              # OCR trên frame / thumbnail
├── asr_processor.py              # speech-to-text
└── text_cleaner.py               # regex cleaning, lowercasing, remove html, etc.
```

Output:

* Chuỗi text (ASR/OCR) gắn với video_id
* Có thể đưa vào Phobert encoder (reuse code từ `metadata/text_embeddings.py`)

---

## 5️⃣ `features/` — Build & Save multimodal features

```text
features/
├── __init__.py
├── multimodal_feature_builder.py # ghép video_emb + audio_emb + metadata_emb + text_emb
├── feature_saver.py              # lưu ra parquet / JSON / NPY / MinIO
└── feature_schema.py             # schema cho 1 sample multimodal
```

* `multimodal_feature_builder.py`:

  * Nhận input từ các pipeline:

    * `metadata_preprocessor` → numeric + desc/tags/comments emb
    * `video_encoder` → video_emb
    * `audio_encoder` → audio_emb
    * (optional) OCR/ASR emb
  * Gộp lại (concat / projection / pooling) → `multimodal_feature`

* `feature_saver.py`:

  * Lưu mỗi sample hoặc batch ra:

    * Local: `.jsonl`, `.parquet`, `.npy`, `.pt`
    * MinIO: theo layout Medallion: `silver/` hoặc `gold/`

---

## 6️⃣ `utils/` — Helper chung

```text
utils/
├── __init__.py
├── file_io.py                    # đọc/ghi JSON, CSV, parquet, NPY
├── minio_utils.py                # client kết nối MinIO (list, get, put)
├── logging_utils.py              # logger thống nhất cho pipeline
├── timer.py                      # context manager đo thời gian
└── constants.py                  # đường dẫn, tên bucket, key chuẩn
```

Ví dụ patterns:

* `file_io.py`:

  * `load_json(path)`, `save_json(obj, path)`, `save_parquet(df, path)`, …

* `minio_utils.py`:

  * `get_minio_client()`, `download_from_minio(bucket, key, local_path)`, …

---

## 7️⃣ `pipelines/` — Orchestrate từng bước

```text
pipelines/
├── __init__.py
├── preprocess_metadata_pipeline.py   # chạy MetadataPreprocessor cho list video
├── preprocess_video_pipeline.py      # xử lý toàn bộ video → video_emb
├── preprocess_audio_pipeline.py      # xử lý toàn bộ video → audio_emb
└── build_multimodal_dataset.py       # join tất cả feature lại thành dataset training
```

### `preprocess_metadata_pipeline.py`

* Đọc file manifest (danh sách video + đường dẫn metadata JSON)
* Với mỗi video:

  * Load metadata JSON
  * Gọi `MetadataPreprocessor.transform_single(meta)`
  * Lưu kết quả: numeric_scaled + embeddings vào Silver layer

### `preprocess_video_pipeline.py`

* Đọc danh sách video_s3_path / local_path
* Extract frames / clip, encode → `video_emb`
* Lưu embedding (theo video_id)

### `preprocess_audio_pipeline.py`

* Extract audio từ video
* Encode → `audio_emb`
* Lưu embedding (theo video_id)

### `build_multimodal_dataset.py`

* Join metadata_emb + video_emb + audio_emb (+ OCR/ASR nếu có) theo `video_id`
* Tạo final dataset (Parquet/JSONL) để model training đọc vào.

---

## 8️⃣ `notebooks/` — Debug & EDA

```text
notebooks/
├── debug_metadata.ipynb      # test MetadataPreprocessor trên vài JSON mẫu
├── debug_comments.ipynb      # visualize toxicity / constructiveness trong comment
├── test_video_embedding.ipynb# test video encoder trên 1–2 video
└── EDA_metadata.ipynb        # EDA phân phối likes, shares, age_days, ...
```

Dùng để:

* Kiểm tra xem feature đã hợp lý chưa
* Vẽ histogram / scatter / PCA trên embedding
* Debug nhanh mà không cần chạy full pipeline

---

## 🔄 Luồng chạy cơ bản

1. Chuẩn bị danh sách video (manifest) chứa:

   * `video_id`
   * `metadata_path` (JSON)
   * `video_s3_path` hoặc `video_local_path`

2. Chạy từng pipeline:

```bash
# 1) Metadata
python offline-training/preprocessing/pipelines/preprocess_metadata_pipeline.py

# 2) Video
python offline-training/preprocessing/pipelines/preprocess_video_pipeline.py

# 3) Audio
python offline-training/preprocessing/pipelines/preprocess_audio_pipeline.py

# 4) Build multimodal dataset (join tất cả lại)
python offline-training/preprocessing/pipelines/build_multimodal_dataset.py
```

3. Output cuối:

   * Một file/bộ file Parquet/JSONL trong `offline-training/datasets/` hoặc MinIO `tikharm/silver` / `tikharm/gold`
   * Dùng trực tiếp cho `training/` (DataLoader + Trainer).
