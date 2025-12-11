
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

# Config
# Should match what simulation_pipeline.py uses
DATA_FILE = Path(__file__).resolve().parents[3] / "data-pipeline/simulated_predictions.jsonl"

def _load_data() -> List[Dict]:
    """Helper to load all data from JSONL"""
    data = []
    if not DATA_FILE.exists():
        return []
        
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        # Normalize timestamp
                        if "timestamp" in record:
                            # Convert ISO string to datetime
                            try:
                                record["ingested_at"] = datetime.fromisoformat(record["timestamp"])
                            except:
                                record["ingested_at"] = datetime.now()
                        else:
                             record["ingested_at"] = datetime.now()
                        data.append(record)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Error reading data file: {e}")
        return []
    return data

def get_recent_predictions(limit: int = 50, label_filter: str = "All") -> List[Dict]:
    """
    Fetch recent predictions.
    """
    all_data = _load_data()
    
    # Filter
    if label_filter != "All":
        all_data = [d for d in all_data if d.get("label") == label_filter]
        
    # Sort desc by digested_at
    all_data.sort(key=lambda x: x["ingested_at"], reverse=True)
    
    return all_data[:limit]

def get_stats(time_range_start: datetime = None, time_range_end: datetime = None) -> Dict:
    """
    Compute aggregate stats.
    """
    all_data = _load_data()
    
    # Filter by time
    if time_range_start and time_range_end:
        all_data = [d for d in all_data if time_range_start <= d["ingested_at"] <= time_range_end]

    stats = {
        "total": 0,
        "counts": {}
    }
    
    stats["total"] = len(all_data)
    
    counts = {}
    for d in all_data:
        label = d.get("label", "Unknown")
        counts[label] = counts.get(label, 0) + 1
        
    stats["counts"] = counts
    return stats

def get_time_series_data(limit_hours: int = 24):
    """
    Get prediction counts per hour/minute.
    """
    # Simply return DF of recently loaded data
    all_data = _load_data()
    if not all_data:
        return pd.DataFrame()

    # Sort
    all_data.sort(key=lambda x: x["ingested_at"], reverse=True)
    
    # Limit somewhat to imitate "recent" query if file is huge
    # For simulation, just take last 2000
    recent = all_data[:2000]
    
    df = pd.DataFrame(recent)
    return df
