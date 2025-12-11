# 🏠 Hướng dẫn Chạy Crawler trên Máy Cá Nhân (Local Host)

Do TikTok chặn IP của Server, bạn cần chạy Crawler trên máy tính cá nhân (Laptop/PC) để lấy dữ liệu thật, sau đó đẩy lên Server qua SSH Tunnel.

## 🛠 Bước 1: Chuẩn bị Môi trường (trên Máy Cá Nhân)

1.  **Cài đặt Python 3.9+** (Nếu chưa có).
2.  **Pull code về máy**:
    ```bash
    git pull origin main
    # Hoặc clone mới nếu chưa có
    git clone https://github.com/lamnguyentuan/IE212-BigData.git
    cd IE212-BigData
    ```
3.  **Tạo môi trường ảo & Cài thư viện**:
    ```bash
    python -m venv venv
    
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate

    pip install -r requirements-crawl.txt
    playwright install chromium
    ```

## 🔗 Bước 2: Tạo SSH Tunnel tới Server

Mở một terminal **KHÁC**, chạy lệnh sau để kết nối MinIO trên Server về máy bạn:

```bash
# Thay username và IP bằng thông tin Server của bạn
ssh -L 9009:localhost:9009 -L 9000:localhost:9000 ezycloudx-admin@<SERVER_IP>
```
*Giữ nguyên cửa sổ này trong suốt quá trình chạy.*

## 🏃 Bước 3: Chạy Crawler

Mở lại terminal (đã active venv), chạy lệnh sau:

**Windows (PowerShell):**
```powershell
$env:MINIO_ENDPOINT="localhost:9009"
$env:MINIO_ACCESS_KEY="minioadmin"
$env:MINIO_SECRET_KEY="minioadmin"
$env:MINIO_BUCKET="tiktok-realtime"

python demo-crawl.py
```

**Mac/Linux:**
```bash
export MINIO_ENDPOINT="localhost:9009"
export MINIO_ACCESS_KEY="minioadmin"
export MINIO_SECRET_KEY="minioadmin"
export MINIO_BUCKET="tiktok-realtime"

python demo-crawl.py
```

## ✅ Kết quả

1.  Trình duyệt Chromium sẽ tự bật lên và vào TikTok (đừng tắt nó).
2.  Sau khi chạy xong, vào MinIO Console (`localhost:9001`) kiểm tra bucket `tiktok-realtime`.
3.  Quay lại Server, chạy tiếp **Step 2 (Producer)** & **Step 3 (Spark)** như kịch bản cũ!
