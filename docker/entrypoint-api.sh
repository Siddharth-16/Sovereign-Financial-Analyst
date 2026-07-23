#!/usr/bin/env sh
set -e

python scripts/sync_chroma.py

exec uvicorn api.main:app --host 0.0.0.0 --port 8000
