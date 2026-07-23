#!/usr/bin/env sh
set -e

python scripts/sync_chroma.py

exec streamlit run ui/ui.py --server.port=8501 --server.address=0.0.0.0