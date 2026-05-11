# Stage 1: Build dependencies and download assets
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Download models and NLTK data
COPY download_assets.py .
ENV NLTK_DATA=/build/nltk_data
ENV SENTENCE_TRANSFORMERS_HOME=/build/models
ENV HUGGINGFACE_HUB_CACHE=/build/models/hub

RUN mkdir -p $NLTK_DATA $SENTENCE_TRANSFORMERS_HOME
RUN python download_assets.py

# Stage 2: Final runtime image
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies (libgomp1 is required for faiss-cpu, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy pre-downloaded models and NLTK data
# We set the environment variables to point to these locations
COPY --from=builder /build/nltk_data /app/nltk_data
COPY --from=builder /build/models /app/models

# Copy application code and data indices
COPY src/ ./src/
COPY data/indices/ ./data/indices/

# Set environment variables
ENV NLTK_DATA=/app/nltk_data
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
ENV HUGGINGFACE_HUB_CACHE=/app/models/hub
ENV PYTHONUNBUFFERED=1

# Expose the FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
