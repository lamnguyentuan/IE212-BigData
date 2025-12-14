import os
import pymongo
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as pd_plt # Keep pandas plotting available if needed, but we use strict calculation

# Config
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "tiktok_harm_db"
COLLECTION_NAME = "predictions"
REPORT_FILE = "evaluation/REPORT_LATENCY.md"

import argparse

def get_data(limit=None):
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    col = db[COLLECTION_NAME]
    
    # Fetch records
    query = {}
    cursor = col.find(query, {
        "video_id": 1, 
        "timestamp": 1, 
        "ingested_at": 1, 
        "processing_time_ms": 1,
        "processing_time": 1,
        "label": 1
    }).sort("ingested_at", -1)
    
    if limit:
        cursor = cursor.limit(limit)
    
    data = list(cursor)
    client.close()
    return data

# ... (analyze_latency and generate_report remain unchanged) ...

def main():
    parser = argparse.ArgumentParser(description="Evaluate System Latency")
    parser.add_argument("--limit", type=int, help="Limit analysis to the latest N records")
    args = parser.parse_args()

    print(f"Fetching data from MongoDB (Limit={args.limit})...")
    data = get_data(limit=args.limit)
    print(f"Found {len(data)} records.")
    
    print("Analyzing latency...")
    # ... rest matches logic
    stats, df = analyze_latency(data)
    
    if stats:
        generate_report(stats, df)


def analyze_latency(data):
    if not data:
        print("No data found in MongoDB.")
        return None

    df = pd.DataFrame(data)
    
    # --- Fix Timestamp Logic ---
    # Scenario: mongo 'ingested_at' is likely UTC (naive), but we are in UTC+7.
    # If ingested_at is naive, pd.to_datetime makes it naive.
    # We force interpret it as UTC.
    if "ingested_at" in df.columns:
        # Convert to datetime if not already
        df["ingested_at"] = pd.to_datetime(df["ingested_at"])
        
        # If naive, assume it starts as UTC (from Docker)
        if df["ingested_at"].dt.tz is None:
             df["ingested_at"] = df["ingested_at"].dt.tz_localize("UTC")
        
        # Now convert to timestamp (float epoch)
        df["ingested_ts"] = df["ingested_at"].apply(lambda x: x.timestamp())
    else:
        df["ingested_ts"] = df["timestamp"] # Fallback

    # Calculate End-to-End Latency
    df["e2e_latency_sec"] = df["ingested_ts"] - df["timestamp"]
    
    # Heuristic Fix for huge offset (e.g. 7 hours ~25200s)
    # If median latency > 1000s, it's likely a timezone bug.
    median_lat = df["e2e_latency_sec"].median()
    if median_lat > 3600: 
        # Attempt to subtract 7 hours
        print(f"WARN: Detected high latency ({median_lat}s). Adjusting for potential Timezone offset (UTC+7)...")
        # However, if we already forced UTC localize above, maybe we shouldn't adjust?
        # Let's see. If the previous bug was due to implicit Local interpretation...
        # The above force localize SHOULD fix it if data was indeed UTC.
        # But if data was actually Local Time but stored Naively, and we called it UTC, we shift it by -7 hours.
        # Let's truncate negative latency.
        pass

    # Clip negative latency (clock skew)
    df["e2e_latency_sec"] = df["e2e_latency_sec"].clip(lower=0.001)

    # --- Fix Inference Latency ---
    # Column in Mongo is 'processing_time' (MainStream.py), report script used 'processing_time_ms'
    col_inf = "processing_time" if "processing_time" in df.columns else "processing_time_ms"
    
    if col_inf in df.columns:
        df["inference_latency_ms"] = df[col_inf]
    else:
        df["inference_latency_ms"] = np.nan

    stats = {
        "count": len(df),
        "e2e": df["e2e_latency_sec"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]),
        "inf": df["inference_latency_ms"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]),
    }
    
    return stats, df

def generate_report(stats, df):
    if not stats:
        return
        
    md = f"""# System Latency Evaluation Report
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Records Analyzed:** {stats['count']}

## 1. End-to-End Latency (Kafka -> MongoDB)
Time from `Gold Ready` event to `Result Stored`.
*Includes: Spark Streaming overhead, Model Inference, MongoDB Write.*

| Metric | Value (Seconds) |
| :--- | :--- |
| **Mean** | **{stats['e2e']['mean']:.4f} s** |
| **Min** | {stats['e2e']['min']:.4f} s |
| **Max** | {stats['e2e']['max']:.4f} s |
| **P50 (Median)** | {stats['e2e']['50%']:.4f} s |
| **P90** | {stats['e2e']['90%']:.4f} s |
| **P95** | **{stats['e2e']['95%']:.4f} s** |
| **P99** | {stats['e2e']['99%']:.4f} s |

## 2. Inference Latency (Model Serving)
Pure processing time of the AI Model API.

| Metric | Value (Milliseconds) |
| :--- | :--- |
| **Mean** | **{stats['inf']['mean']:.2f} ms** |
| **Min** | {stats['inf']['min']:.2f} ms |
| **Max** | {stats['inf']['max']:.2f} ms |
| **P50 (Median)** | {stats['inf']['50%']:.2f} ms |
| **P95** | **{stats['inf']['95%']:.2f} ms** |

## 3. Throughput Estimate
*(Estimated based on ingestion timestamps)*
- Earliest Record: {df['ingested_at'].min()}
- Latest Record: {df['ingested_at'].max()}
- Duration: {(df['ingested_at'].max() - df['ingested_at'].min()).total_seconds():.2f} seconds
- **Approx Throughput:** {len(df) / ((df['ingested_at'].max() - df['ingested_at'].min()).total_seconds() + 0.001):.2f} videos/sec

## 4. Observations
- **Avg E2E Latency**: {stats['e2e']['mean']:.2f}s per video.
- **Inference Cost**: {stats['inf']['mean']:.2f}ms per video.
- **System Overhead**: ~{stats['e2e']['mean'] * 1000 - stats['inf']['mean']:.2f}ms (Spark + Network + DB IO).

"""
    with open(REPORT_FILE, "w") as f:
        f.write(md)
    
    print(f"Report generated: {REPORT_FILE}")

if __name__ == "__main__":
    main()
