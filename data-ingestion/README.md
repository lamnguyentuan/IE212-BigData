# Data Ingestion Module

This module handles the acquisition of video data and metadata from various sources and their ingestion into the **Bronze Layer** of our MinIO Data Lake.

## Components

### 1. TikTok Crawler (`tiktok_crawl/`)
A generic scraper using `Playwright` to fetch videos from TikTok based on hashtags or user profiles.
- **Output**: Downloads video (`.mp4`) and extracts metadata (`.json`).
- **Destinations**: Local disk or MinIO Bronze bucket.

### 2. TikHarm Uploader (`tikharm_upload/`)
A specialized utility to ingest a pre-existing dataset (TikHarm) consisting of categorized harmful/safe videos.
- **Function**: Reads a standard dataset directory structure and uploads files to MinIO, maintaining the folder structure in the object store.
- **Manifest**: Generates a `dataset_manifest.json` for validation.

## Usage

See the specific README files in each subdirectory for detailed usage instructions.
