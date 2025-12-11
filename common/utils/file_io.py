from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, List

import json
import numpy as np
import yaml
import shutil
import logging

from pathlib import Path

logger = logging.getLogger("file-io")

def safe_rmtree(path: Path) -> None:
    """
    Xoá 1 thư mục và toàn bộ nội dung bên trong (nếu tồn tại).
    Dùng cho việc clean local sau khi xử lý 1 video.
    """
    if path.exists():
        logger.info(f"[file-io] Removing directory: {path}")
        shutil.rmtree(path, ignore_errors=True)


def ensure_dir(path: Path) -> Path:
    """Tạo thư mục nếu chưa tồn tại. Trả lại chính path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    """Đọc JSON UTF-8."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any, indent: int = 2) -> None:
    """Ghi object thành JSON UTF-8, pretty indent."""
    ensure_dir(path.parent)
    text = json.dumps(obj, ensure_ascii=False, indent=indent)
    path.write_text(text, encoding="utf-8")


def read_yaml(path: Path) -> Dict[str, Any]:
    """Đọc YAML thành dict. Nếu file không tồn tại → {}."""
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw or {}


def write_yaml(path: Path, obj: Dict[str, Any]) -> None:
    """Ghi dict thành YAML."""
    ensure_dir(path.parent)
    text = yaml.safe_dump(obj, allow_unicode=True, sort_keys=False)
    path.write_text(text, encoding="utf-8")


def list_subdirs(path: Path) -> List[Path]:
    """Liệt kê các thư mục con trực tiếp của path."""
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_dir()])


def save_npz(path: Path, **arrays: Any) -> None:
    """Ghi nhiều mảng numpy vào 1 file .npz."""
    ensure_dir(path.parent)
    np.savez_compressed(path, **arrays)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    """Đọc file .npz thành dict tên → np.ndarray."""
    with np.load(path, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}
