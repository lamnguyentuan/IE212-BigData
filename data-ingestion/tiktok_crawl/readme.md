# 📦 TikTok Crawler & Ingestion

Included in `tiktok_crawl/`:
- `tiktok_scraper.py`: Main scraper + MinIO uploader.
- `config.yaml`: Configuration.

## 📥 Metadata JSON Structure
Saved to `tiktok-data/bronze/{video_id}/metadata.json`.

```json
{
  "video_id": "723...",
  "url": "https://www.tiktok.com/...",
  "timestamp": 167...,
  "source": "tiktok_crawl",
  "stats": {
      "likes": 100,
      "comments": 5,
      "shares": 1
  },
  "description": "Video caption..."
}
```

## 🛠 Usage
```bash
# Run Scraper
python tiktok_scraper.py
```

It auto-checks `downloaded_videos.txt` to avoid duplicates.
