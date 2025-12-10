"""
Configuration Loader Utility.

Provides helpers to load YAML files and environment variables uniformly.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

def load_yaml(path: str) -> Dict[str, Any]:
    """
    Load a YAML file safely.
    
    Args:
        path: Path to the yaml file.
    
    Returns:
        Dictionary containing the yaml content.
    
    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_env_or_default(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get environment variable or return default.
    """
    return os.getenv(key, default)
