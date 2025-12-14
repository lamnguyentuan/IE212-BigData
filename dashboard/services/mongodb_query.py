"""
MongoDB Query Service.

Fetches analytics and recent alerts from MongoDB.
"""

from pymongo import MongoClient
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict

import os

# Config - Ideally load from env or central config
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "tiktok_harm_db"
COLLECTION_NAME = "predictions"

def get_client():
    return MongoClient(MONGO_URI)

def get_recent_predictions(limit: int = 50, label_filter: str = "All") -> List[Dict]:
    """
    Fetch recent predictions.
    """
    client = get_client()
    db = client[DB_NAME]
    col = db[COLLECTION_NAME]
    
    query = {}
    if label_filter != "All":
        query["label"] = label_filter
        
    cursor = col.find(query).sort("timestamp", -1).limit(limit)
    
    data = list(cursor)
    client.close()
    return data

def get_stats(time_range_start: datetime = None, time_range_end: datetime = None) -> Dict:
    """
    Compute aggregate stats.
    """
    client = get_client()
    db = client[DB_NAME]
    col = db[COLLECTION_NAME]
    
    match_stage = {}
    if time_range_start and time_range_end:
        match_stage["ingested_at"] = {"$gte": time_range_start, "$lte": time_range_end}
        
    pipeline = [
        {"$match": match_stage},
        {"$group": {"_id": "$label", "count": {"$sum": 1}}}
    ]
    
    results = list(col.aggregate(pipeline))
    client.close()
    
    stats = {
        "total": 0,
        "counts": {}
    }
    
    for r in results:
        label = r["_id"] or "Unknown"
        count = r["count"]
        stats["counts"][label] = count
        stats["total"] += count
        
    return stats

def get_time_series_data(limit_hours: int = 24):
    """
    Get prediction counts per hour/minute.
    """
    client = get_client()
    db = client[DB_NAME]
    col = db[COLLECTION_NAME]
    
    # Simple fetch and pandas resample for dashboard flexibility
    # Optimization: perform aggregation in Mongo for specific intervals
    
    # Returning raw DF for streamlet chart handling
    cursor = col.find({}, {"ingested_at": 1, "label": 1}).sort("ingested_at", -1).limit(2000)
    data = list(cursor)
    client.close()
    
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data)
    if "ingested_at" in df.columns:
        df["ingested_at"] = pd.to_datetime(df["ingested_at"])
    return df
