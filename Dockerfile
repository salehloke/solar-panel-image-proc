# Base image with Python and minimal OS libs
FROM python:3.11-slim

# Install system deps for OpenCV and image libs
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       git \
       libgl1 \
       libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create work dir
WORKDIR /app

# Copy Python requirements first to leverage Docker cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the repo
COPY . .

# Default command; override in docker-compose
ENTRYPOINT ["python"]
CMD ["src/train.py", "--data_dir", "data/processed"]
