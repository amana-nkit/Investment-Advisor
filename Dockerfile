# ── Stage 1: build ──────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt --target /build/packages

# ── Stage 2: runtime ─────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Non-root user
RUN addgroup --system app && adduser --system --ingroup app app

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /build/packages /usr/local/lib/python3.12/site-packages

# Copy application code
COPY --chown=app:app . .

# Runtime directories
RUN mkdir -p logs chroma_db \
 && chown -R app:app logs chroma_db

USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "ui/app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
