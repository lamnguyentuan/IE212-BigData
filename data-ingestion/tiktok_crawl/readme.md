# 📦 TikTok Crawler – Big Data Ingestion Module

Thư mục **`tiktok_crawl/`** chứa toàn bộ mã nguồn phục vụ **thu thập dữ liệu TikTok** và lưu trữ lên **MinIO Data Lake** theo kiến trúc **Medallion (Bronze / Silver / Gold)**.
Đây là bước đầu tiên của pipeline Big Data trong dự án IE212 — tạo nguồn dữ liệu cho giai đoạn xử lý, tiền xử lý và huấn luyện mô hình.

---

# ⚙️ 1. Chức năng chính

Module TikTok Crawler gồm:

### ✔ Crawl video theo hashtag

* Thu thập danh sách video từ trang hashtag TikTok
* Cho phép chỉnh số lượng video mỗi hashtag
* Tự động mở CAPTCHA nếu TikTok yêu cầu xác minh

### ✔ Trích xuất metadata video

Bao gồm:

* lượt thích
* lượt bình luận
* lượt chia sẻ
* mô tả video
* danh sách hashtag
* ngày đăng
* tác giả
* lượt xem

### ✔ Crawl cây bình luận (comments tree)

* `comment-level-1`: bình luận gốc
* `comment-level-2`: các reply của bình luận tương ứng
* Tự động mở panel comment
* Tự động click toàn bộ nút "View replies"

### ✔ Tải video (tuỳ chọn) với `yt_dlp`

### ✔ Upload video + metadata lên MinIO

Theo chuẩn Medallion Architecture:

```
tiktok-data/
└── bronze/
    └── {video_id}/
        ├── video.mp4
        └── metadata.json
```

---

# 📁 2. Cấu trúc thư mục

```
tiktok_crawl/
├── README.md
├── __init__.py               # Export class TikTokScraper
├── config.yaml               # File config cho toàn bộ crawler
└── tiktok_scraper.py         # Mã nguồn chính của crawler
```

---

# ⚙️ 3. File cấu hình: `config.yaml`

Tất cả thông số được chỉnh tại 1 file duy nhất:

```yaml
timeout_ms: 10000
viewport_width: 1280
viewport_height: 720

# Crawl behavior
max_comments_rounds: 40
min_sleep_sec: 1.0
max_sleep_sec: 3.0
download_video: true

# Hashtag settings
hashtags:
  - xuhuong
  - fyp
max_videos_per_hashtag: 10

# MinIO integration
minio_config: "config_tiktok.yaml"

# Local paths
export_dir: "exports"
log_file: "logs/tiktok_scraper.log"
tmp_download_dir: "tmp_downloads"
```

### Ưu điểm:

* Không cần sửa Python khi muốn đổi hashtag
* Có thể bật/tắt download video
* Dễ chỉnh timeout, viewport, số vòng scroll để load comment
* Dễ đổi bucket MinIO hoặc đường dẫn

---

# 🧠 4. File chính: `tiktok_scraper.py`

Chứa class:

```
TikTokScraper
```

Chịu trách nhiệm:

* Load config từ `config.yaml`
* Điều khiển Playwright
* Trích xuất metadata
* Crawl comment tree
* Tải video bằng yt_dlp
* Upload video / metadata JSON lên MinIO bằng `minio_client.py`
* Lưu bản sao metadata ra thư mục `exports/`

### Một số hàm quan trọng:

| Hàm                                | Mô tả                            |
| ---------------------------------- | -------------------------------- |
| `_extract_video_info()`            | Lấy metadata + comment tree      |
| `_download_video()`                | tải video bằng yt_dlp            |
| `_upload_file_to_minio()`          | upload file video lên MinIO      |
| `_save_json_to_minio()`            | upload metadata JSON lên MinIO   |
| `_collect_video_urls_by_hashtag()` | crawl URL video từ trang hashtag |
| `scrape_hashtag()`                 | crawl nhiều video theo hashtag   |
| `scrape_single_video()`            | crawl 1 video cụ thể             |
| `save_results_local()`             | lưu kết quả ra file JSON         |

---

# 🗄 5. Cách chạy

Từ root dự án:

```bash
python data-ingestion/tiktok_crawl/tiktok_scraper.py
```

Crawler sẽ:

1. Đọc danh sách hashtag trong `config.yaml`
2. Mỗi hashtag crawl tối đa `max_videos_per_hashtag`
3. Lưu video & metadata vào MinIO
4. Xuất file demo JSON vào `exports/`
5. Lưu log vào `logs/tiktok_scraper.log`

---

# 🪣 6. Dữ liệu được đẩy vào MinIO như thế nào?

Ví dụ video có ID:

```
1234567890
```

Crawler upload:

### Video MP4:

```
tiktok-data/bronze/1234567890/video.mp4
```

### Metadata JSON:

```
tiktok-data/bronze/1234567890/metadata.json
```

Đúng chuẩn **Bronze Layer** trong kiến trúc Medallion.

---

# 🔌 7. Tích hợp MinIO

Crawler sử dụng:

```
minio/minio_client.py
```

với config file:

```
minio/config_tiktok.yaml
```

Chỉ cần thay đổi `minio_config` trong `config.yaml` là có thể dùng bucket khác.

---

# 🧪 8. Test 1 video cụ thể (tuỳ chọn)

Trong `tiktok_scraper.py`, bạn có thể bật:

```python
video_info = await scraper.scrape_single_video("https://www.tiktok.com/@someone/video/123...")
```
