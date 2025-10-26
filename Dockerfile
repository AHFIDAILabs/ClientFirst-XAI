# Multi-stage build for optimized image size
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=10

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata first (leverages Docker caching)
COPY setup.py README.md requirements.txt ./
COPY src/ src/
# ✅ Add this line to include model artifacts
COPY artifacts/ artifacts/

# Install dependencies
RUN pip install --user -r requirements.txt


# ===== Final runtime image =====
FROM python:3.11-slim
WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy all app code and model artifacts
COPY app/ app/
COPY config/ config/
COPY src/ src/
# ✅ Copy artifacts into runtime image
COPY --from=builder /app/artifacts /app/artifacts

# Create logs directory
RUN mkdir -p logs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]