# Minimal production image for FastAPI backend
FROM python:3.11-slim

# System deps (optional: curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install deps first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt httpx==0.27.2 pyyaml==6.0.2

# Copy app code
COPY . .

# Non-root (optional)
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Env hints
ENV PORT=8000 \
    CORS_ORIGINS=* \
    TORCH_FILTER_URL=http://localhost:9000 \
    FILTER_BLOCK=0.97 \
    FILTER_ALLOW=0.03 \
    FILTER_EARLY_ALLOW=0

EXPOSE 8000

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
