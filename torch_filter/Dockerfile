# Minimal image for the PyTorch filter microservice
FROM python:3.11-slim

# System deps for Pillow and tokenizers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libjpeg62-turbo-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy & install Python deps first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY . .

# Create filters/ if missing (TorchScript models go here)
RUN mkdir -p filters

# Non-root user (optional, safer)
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Env hints (override at runtime if needed)
ENV IMAGE_TS=filters/model.ts \
    TEXT_TS=filters/text_model.ts \
    TEXT_MODEL=distilroberta-base \
    MAXLEN=256

EXPOSE 9000

# Start FastAPI service
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "9000"]
