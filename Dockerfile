FROM python:3.11-slim AS base

# ── System deps ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps (cached layer) ────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────
COPY src/      ./src/
COPY scripts/  ./scripts/

# ── Healthcheck ───────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -sf http://localhost:8000/api/v1/health || exit 1

# ── Ports ─────────────────────────────────────────────────────────────
EXPOSE 8000   # FastAPI
EXPOSE 8501   # Streamlit

# ── Entrypoint ────────────────────────────────────────────────────────
# Runs both services in a single container.
# In production, split into separate containers via docker-compose or K8s.
CMD ["bash", "-c", \
     "uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2 & \
      streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0 & \
      wait"]
