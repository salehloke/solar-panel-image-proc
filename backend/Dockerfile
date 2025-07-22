# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

RUN echo "Model download failed, will download at runtime"
# Pre-download models during build (optional)
RUN python scripts/download_model.py || echo "Model download failed, will download at runtime"

# Create non-root user and set up cache directory
RUN adduser --disabled-password --gecos '' appuser \
    && mkdir -p /home/appuser/.cache/torch/hub \
    && mkdir -p /app/models \
    && chown -R appuser:appuser /app \
    && chown -R appuser:appuser /home/appuser/.cache
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 