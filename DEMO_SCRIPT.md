# 🎬 KỊCH BẢN DEMO ĐỒ ÁN BIG DATA (Real-time Pipeline)

**Dự án**: TikTok Harmfulness Detection System
**Người trình bày**: [Tên bạn]

---

## 🟢 PHẦN 0: CHUẨN BỊ (Làm trước khi thầy gọi)

1.  **Mở 3 cửa sổ Terminal riêng biệt** (để chạy 3 thành phần của hệ thống).
2.  **Mở trình duyệt với 4 Tabs sau**:
    *   **Dashboard**: `http://localhost:8501` (Trang chính hiển thị kết quả)
    *   **Airflow**: `http://localhost:8081` (Quản lý luồng)
    *   **MinIO**: `http://localhost:9001` -> Vào Bucket `tiktok-realtime`.
    *   **Slide báo cáo** (Nếu có).

---

## 🎬 PHẦN 1: THU THẬP DỮ LIỆU (Ingestion)
*Mục tiêu: Chứng minh hệ thống lấy được dữ liệu thật từ TikTok.*

**🗣 Lời dẫn**:
*"Thưa thầy, hệ thống bắt đầu bằng việc thu thập dữ liệu thời gian thực. Em sẽ kích hoạt Crawler để lấy các video mới nhất theo từ khóa `#review` và lưu vào Bucket demo riêng là `tiktok-realtime`."*

**Youtube/Terminal 1 (Chạy Crawler)**:
```bash
docker exec -e MINIO_BUCKET="tiktok-realtime" crawler python data_pipeline/crawl_only.py review
```

*   **Hành động**: Chuyển ngay sang Tab **MinIO**.
*   **Quan sát**: Sau khoảng 10-15s, thầy sẽ thấy các folder video mới xuất hiện trong bucket `tiktok-realtime/bronze`.
*   **Chốt**: *"Dữ liệu thô (Video MP4 + Metadata) đã được tải thành công về Data Lake."*

---

## 🚀 PHẦN 2: XỬ LÝ & PHÂN TÍCH (Processing)
*Mục tiêu: Chứng minh dữ liệu chảy qua Kafka và được Spark + AI Model xử lý.*

**🗣 Lời dẫn**:
*"Ngay khi có video mới, hệ thống sẽ đẩy sự kiện vào Kafka. Spark Streaming sẽ đón nhận luồng dữ liệu này để xử lý và gọi AI Model dự đoán xem video có độc hại hay không."*

**Terminal 2 (Chạy Producer - Giả lập sự kiện từ Crawler)**:
```bash
export MINIO_ENDPOINT="localhost:9009"
export MINIO_ACCESS_KEY="minioadmin"
export MINIO_SECRET_KEY="minioadmin"
export MINIO_BUCKET="tiktok-realtime"

venv/bin/python data_pipeline/producer_simulator.py
```

**Terminal 3 (Chạy Spark - Bộ xử lý trung tâm)**:
```bash
export MINIO_ENDPOINT="localhost:9009"
export MINIO_ACCESS_KEY="minioadmin"
export MINIO_SECRET_KEY="minioadmin"
export MONGO_URI="mongodb://localhost:27017/"

venv/bin/python data_pipeline/spark-streaming/main_stream.py --mode stream
```

*   **Quan sát**: Bạn sẽ thấy Logs chạy liên tục ở Terminal 3: `Processing batch...`, `Result: Safe (0.98)...`
*   **Chốt**: *"Spark đang xử lý từng lô dữ liệu (Micro-batch), tích hợp Model Multimodal để đưa ra kết quả phân loại."*

---

## 📊 PHẦN 3: HIỂN THỊ KẾT QUẢ (Dashboard)
*Mục tiêu: Show kết quả End-to-End cho người dùng cuối.*

**🗣 Lời dẫn**:
*"Kết quả phân tích cuối cùng được hiển thị trực quan trên Dashboard quản trị."*

**Hành động**:
1.  Chuyển sang Tab **Dashboard** (`localhost:8501`).
2.  Bấm nút **"Refresh Data"** ở thanh bên trái.
3.  Chỉ vào biểu đồ và bảng **"Recent Alerts"**.
4.  **Click vào một dòng bất kỳ** trong bảng danh sách.
5.  Video Player sẽ hiện ra và phát video đó.

**Chốt**: *"Thầy có thể thấy hệ thống đã phát hiện video này là [Safe/Harmful] với độ tin cậy [X%]. Video được stream trực tiếp từ MinIO Server để kiểm chứng."*

---

## ✅ KẾT THÚC DEMO
*"Đó là toàn bộ luồng dữ liệu End-to-End của nhóm em. Cảm ơn thầy đã theo dõi."*
