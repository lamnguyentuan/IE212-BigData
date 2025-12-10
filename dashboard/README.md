# Analytics Dashboard

A **Streamlit** application designed for real-time monitoring and manual review of the detection system.

## Key Features

1.  **Real-Time Metrics**: Total processed videos and harmful content detection rate.
2.  **Alerts Table**: Live feed of videos flagged as "Not Safe" or "Harmful".
3.  **Video Playback**: Direct integration with MinIO (Bronze Layer) to play video files for verification.
4.  **Charts**: Temporal trends and categorization breakdown.

## Data Sources
- **MongoDB**: Reads real-time prediction logs populated by Spark Streaming.
- **MinIO**: Generates signed URLs for video content access.

## Running Locally

```bash
streamlit run app.py
```
Access at `http://localhost:8501`.
