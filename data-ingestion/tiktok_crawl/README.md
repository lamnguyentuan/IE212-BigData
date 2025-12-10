# TikTok Crawler

A robust scraper built with `Playwright` and `yt-dlp` to collect TikTok videos for analysis.

## Features
- **Headless Browsing**: Uses Playwright to navigate TikTok's dynamic content.
- **Metadata Extraction**: Captures likes, comments, shares, hashtags, and descriptions.
- **MinIO Integration**: Directly uploads crawled content to S3-compatible storage (MinIO).
- **Deduplication**: Checks against existing video IDs in MinIO to avoid redundant work.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install playwright minio yt-dlp
   playwright install chromium
   ```

2. **Configuration**:
   Ensure MinIO credentials (`MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`) are set in your environment or `.env` file.

## Usage

Run the scraper via command line:

```bash
# Basic usage (crawl by hashtag)
python tiktok_scraper.py --hashtag "vietnam" --limit 50 --upload-minio

# Crawl by user
python tiktok_scraper.py --user "tiktok_user" --limit 20
```

## Output Structure (Bronze Layer)
The scraper organizes data in MinIO as follows:

```
tiktok-data/
└── bronze/
    └── {video_id}/
        ├── video.mp4
        └── metadata.json
```
