#!/bin/bash

# SolarAI Model Setup Script
# This script pre-downloads models to speed up container startup

set -e

echo "🚀 Setting up SolarAI Models..."

# Create models directory
mkdir -p models

# Run model download in a temporary container
echo "📥 Downloading PyTorch models..."
docker run --rm \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/backend:/app" \
  python:3.11-slim \
  bash -c "
    cd /app && \
    pip install torch torchvision && \
    python scripts/download_model.py
  "

echo "✅ Models downloaded successfully!"
echo "📁 Models saved to: $(pwd)/models/"
echo ""
echo "🎯 Next steps:"
echo "   1. Start the backend: docker-compose -f docker-compose.backend.yml up -d"
echo "   2. Start the frontend: cd frontend && npm run dev"
echo "   3. Access the app: http://localhost:3000" 