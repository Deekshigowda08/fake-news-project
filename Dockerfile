# VERITAS Multimodal Fake News Detection - Backend Service
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CPU/CUDA wheels
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 || \
    pip install --no-cache-dir torch torchvision

# Copy requirements and install dependencies
COPY ml /app/ml
COPY backend /app/backend
COPY data /app/data

RUN pip install --no-cache-dir \
    "transformers<5.0.0" \
    scikit-learn \
    scipy \
    pandas \
    pydantic \
    fastapi \
    "uvicorn[standard]" \
    sqlmodel \
    httpx \
    python-multipart \
    pillow \
    duckduckgo_search

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
