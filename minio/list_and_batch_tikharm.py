#!/usr/bin/env python3
"""
List objects in the `tikharm` bucket, split into batches and optionally
download each batch with per-file retries and resume/start-batch support.

Usage examples:
  # show count and batches (dry-run)
  python minio/list_and_batch_tikharm.py --dry-run

  # download batches to local dir, 30 per batch
  python minio/list_and_batch_tikharm.py --download --dest ./tikharm_downloads

  # start from batch 5 (0-based)
  python minio/list_and_batch_tikharm.py --download --start-batch 5

This script loads the local `minio_client.py` (not the external package) so
it avoids name collisions with the installed `minio` package.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List
import importlib.util
import sys


def load_local_minio_client() -> object:
    """Dynamically load the repository's `minio_client.py` to avoid
    colliding with the installed `minio` package namespace."""
    here = Path(__file__).resolve().parent
    target = here / "minio_client.py"
    if not target.exists():
        raise FileNotFoundError(f"Local minio_client not found at {target}")

    spec = importlib.util.spec_from_file_location("local_minio_client", str(target))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def chunk_list(items: List[str], size: int) -> List[List[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def safe_fget(client, bucket: str, obj_name: str, dest_path: Path, max_retries: int = 3) -> bool:
    """Download an object with simple retry/backoff. Returns True on success."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while attempt < max_retries:
        try:
            # MinIO client's fget_object(bucket_name, object_name, file_path)
            client.fget_object(bucket, obj_name, str(dest_path))
            return True
        except Exception as e:
            attempt += 1
            wait = 2 ** attempt
            print(f"Error downloading {obj_name} (attempt {attempt}/{max_retries}): {e}. Retrying in {wait}s")
            time.sleep(wait)
    print(f"Failed to download {obj_name} after {max_retries} attempts. Skipping.")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="List and batch tikharm objects from MinIO")
    parser.add_argument("--config", default="config_tikharm.yaml", help="MinIO config filename in minio/ folder")
    parser.add_argument("--batch-size", type=int, default=30, help="Objects per batch")
    parser.add_argument("--dest", default="./tikharm_downloads", help="Destination directory for downloads")
    parser.add_argument("--download", action="store_true", help="Download the objects for each batch")
    parser.add_argument("--dry-run", action="store_true", help="Only list and show batches without downloading")
    parser.add_argument("--start-batch", type=int, default=0, help="Batch index to start from (0-based)")
    parser.add_argument("--save-batches", default="batches.json", help="Save batches to JSON file")
    parser.add_argument("--max-files", type=int, default=0, help="If >0, limit how many files are processed (useful for tests)")
    args = parser.parse_args()

    # load local minio client module
    local = load_local_minio_client()
    get_minio_client = getattr(local, "get_minio_client")

    client, bucket = get_minio_client(args.config)

    print(f"Listing objects in bucket: {bucket}...")
    objs = client.list_objects(bucket, recursive=True)

    names: List[str] = []
    for o in objs:
        # object_name attribute for MinIO SDK object
        try:
            names.append(o.object_name)
        except Exception:
            # defensive: if object has different attr name, try str
            names.append(str(o))

    if args.max_files and args.max_files > 0:
        names = names[: args.max_files]

    total = len(names)
    print(f"Found {total} objects in '{bucket}'")

    batches = chunk_list(names, args.batch_size)
    print(f"Split into {len(batches)} batches of up to {args.batch_size} items")

    # optionally save batches layout
    batches_file = Path(args.save_batches)
    with open(batches_file, "w", encoding="utf-8") as f:
        json.dump(batches, f, indent=2, ensure_ascii=False)
    print(f"Saved batches list to {batches_file}")

    if args.dry_run or not args.download:
        print("Dry run complete. Use --download to fetch batches to disk.")
        return

    dest_root = Path(args.dest)
    dest_root.mkdir(parents=True, exist_ok=True)

    for idx, batch in enumerate(batches):
        if idx < args.start_batch:
            print(f"Skipping batch {idx} (start-batch={args.start_batch})")
            continue

        print(f"Processing batch {idx+1}/{len(batches)} with {len(batch)} objects...")

        for name in batch:
            # create a safe local path using the object name
            dest_path = dest_root / name
            if dest_path.exists():
                print(f"Already exists, skipping: {dest_path}")
                continue

            success = safe_fget(client, bucket, name, dest_path)
            if not success:
                # continue to next file; errors are logged in safe_fget
                continue

        print(f"Finished batch {idx}. Sleeping briefly before next batch...")
        time.sleep(1)

    print("All requested batches processed.")


if __name__ == "__main__":
    main()
