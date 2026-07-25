FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV HF_HOME=/srv/.cache/huggingface
ARG EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBED_MODEL}')"

ENV HF_HUB_OFFLINE=1

COPY app ./app
COPY api ./api
COPY ui ./ui
COPY scripts ./scripts
COPY docker ./docker
RUN chmod +x ./docker/entrypoint-api.sh ./docker/entrypoint-ui.sh

RUN useradd --create-home --uid 1000 appuser
RUN mkdir -p /srv/chroma_db && chown -R appuser:appuser /srv
USER appuser

# -------------------------------------------------------------------- ui ---
FROM base AS ui

COPY --chown=appuser:appuser chroma_db* ./chroma_db/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["./docker/entrypoint-ui.sh"]

FROM base AS api

COPY --chown=appuser:appuser chroma_db* ./chroma_db/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["./docker/entrypoint-api.sh"]
