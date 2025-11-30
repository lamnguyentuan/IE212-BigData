# 📂 Local TikHarm Layout & Upload Strategy

Tài liệu này mô tả **cách tổ chức TikHarm trên máy local** và **cách upload trực tiếp lên MinIO** trong module `tikharm_upload/`.

---

## 1. Vị trí TikHarm trên máy local

Module `upload_tikharm_to_minio.py` giả định TikHarm được đặt tại:

```text
offline-training/datasets/TikHarm/
```

Từ root project:

```text
IE212-BigData/
└── offline-training/
    └── datasets/
        └── TikHarm/
            ├── train/
            ├── val/
            └── test/
```

---

## 2. Cấu trúc thư mục TikHarm gốc (raw)

Trong mỗi split (`train/`, `val/`, `test/`), dữ liệu được chia theo **label folder**:

```text
TikHarm/
├── train/
│   ├── Adult Content/
│   ├── Harmful Content/
│   ├── Safe/
│   └── Suicide/
├── val/
│   ├── Adult Content/
│   ├── Harmful Content/
│   ├── Safe/
│   └── Suicide/
└── test/
    ├── Adult Content/
    ├── Harmful Content/
    ├── Safe/
    └── Suicide/
```

Bên trong mỗi thư mục label là **các file video rời rạc**, tên file không theo quy luật:

```text
TikHarm/train/Safe/
├── VID_00123.mp4
├── tiktok_2020_safe_abc.mp4
└── ...
```

---

## 3. Quy hoạch lại ID & label khi upload

Khi upload, chúng ta **không tạo bản preprocessed local**, mà:

* Duyệt trực tiếp TikHarm raw
* Sinh ra một `video_id` chuẩn cho từng video
* Upload trực tiếp lên MinIO với layout dạng thư mục

### 3.1. Mapping label

Label thư mục gốc → slug dùng trong `video_id`:

| Thư mục gốc       | `label_slug` |
| ----------------- | ------------ |
| `Adult Content`   | `adult`      |
| `Harmful Content` | `harmful`    |
| `Safe`            | `safe`       |
| `Suicide`         | `suicide`    |

### 3.2. Quy luật đặt `video_id`

Mỗi video có một `video_id` duy nhất:

```text
{split}_{label_slug}_{running_index}
```

Trong đó:

* `split` ∈ {`train`, `val`, `test`}
* `label_slug` ∈ {`adult`, `harmful`, `safe`, `suicide`}
* `running_index`: số thứ tự tăng dần theo từng `(split, label_slug)`, format 6 chữ số

Ví dụ:

```text
train_safe_000001
train_safe_000002
train_harmful_000001
val_adult_000010
test_suicide_000123
```

---

## 4. Layout trên MinIO (bronze layer)

Script `upload_tikharm_to_minio.py` upload trực tiếp lên bucket (ví dụ: `tikharm`) theo cấu trúc:

```text
tikharm/
└── bronze/
    ├── train_safe_000001/
    │   ├── video.mp4
    │   └── metadata.json
    ├── train_safe_000002/
    │   ├── video.mp4
    │   └── metadata.json
    ├── train_harmful_000001/
    │   ├── video.mp4
    │   └── metadata.json
    ├── val_adult_000010/
    │   ├── video.mp4
    │   └── metadata.json
    └── ...
```

### 4.1. `video.mp4`

* Copy từ file gốc
* Được chuẩn hoá tên thành đúng **`video.mp4`** trong folder mỗi `video_id`
* Định dạng mime tự động đoán qua `mimetypes.guess_type`

### 4.2. `metadata.json`

Mỗi video đi kèm một file metadata, ví dụ:

```json
{
  "video_id": "train_safe_000001",
  "split": "train",
  "label_raw": "Safe",
  "label": "safe",
  "original_filename": "VID_00123.mp4",
  "original_path": "train/Safe/VID_00123.mp4",
  "source": "TikHarm"
}
```

Ý nghĩa:

* `video_id`: ID duy nhất dùng xuyên suốt pipeline (bronze → silver → gold)
* `split`: train / val / test
* `label_raw`: tên thư mục gốc (hữu ích khi debug, đối chiếu với paper gốc)
* `label`: slug chuẩn hoá, dùng để huấn luyện (adult/harmful/safe/suicide)
* `original_filename`: tên file video ban đầu
* `original_path`: path tương đối so với `TikHarm/` trên local
* `source`: nguồn dataset (`TikHarm`)

---

## 5. Script chịu trách nhiệm upload

File chính:

```text
data-ingestion/tikharm_upload/upload_tikharm_to_minio.py
```

Chức năng:

* Kiểm tra tồn tại `offline-training/datasets/TikHarm/`
* Đọc cấu hình MinIO từ `minio/config_tikharm.yaml`
* Duyệt lần lượt:

  * `train/Adult Content`, `train/Safe`, …
  * `val/...`, `test/...`
* Với mỗi video:

  * Sinh `video_id`
  * Upload:

    * `bronze/{video_id}/video.mp4`
    * `bronze/{video_id}/metadata.json`
* Không tạo thêm file/dataset preprocessed nào trên local

---

## 6. Tóm tắt design choice

* ✅ **Không tốn thêm dung lượng local**
  Không tạo `TikHarm_preprocessed/`, xử lý và upload thẳng lên MinIO.

* ✅ **Mỗi video là một “đơn vị dữ liệu” rõ ràng**
  Tất cả thông tin liên quan (video + metadata) nằm gọn trong `bronze/{video_id}/`.

* ✅ **Tách bạch giữa local layout và logical layout trên MinIO**
  Local vẫn giữ nguyên cấu trúc gốc của TikHarm; MinIO dùng layout tối ưu cho training & Big Data pipeline.

* ✅ **Dễ mở rộng sang Silver/Gold layer**
  Từ `video_id` + `label`, có thể dễ dàng build bảng parquet, training set,… ở silver/gold mà không phụ thuộc tên file ban đầu.

