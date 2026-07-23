from __future__ import annotations
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    bucket = os.getenv("CHROMA_S3_BUCKET")
    if not bucket:
        print("sync_chroma: CHROMA_S3_BUCKET not set, skipping (using baked-in/mounted chroma_db).")
        return 0

    try:
        import boto3
    except ImportError:
        print(
            "sync_chroma: CHROMA_S3_BUCKET is set but boto3 isn't installed. "
            "Add `boto3` to requirements.txt to use S3-backed chroma sync.",
            file=sys.stderr,
        )
        return 1

    from app.config import CHROMA_PATH

    prefix = os.getenv("CHROMA_S3_PREFIX", "chroma_db")
    dest = Path(CHROMA_PATH)
    dest.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel_path = os.path.relpath(key, prefix)
            local_path = dest / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(local_path))
            count += 1

    print(f"sync_chroma: pulled {count} object(s) from s3://{bucket}/{prefix} -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
